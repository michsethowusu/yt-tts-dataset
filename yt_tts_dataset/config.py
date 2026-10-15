"""Config loading, paths and the resume-safe per-episode loop shared by every stage."""
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DEFAULTS = {
    "data_dir": "data",
    "source": {"urls": [], "urls_file": "", "min_minutes": 0, "max_minutes": 0},
    "download": {"backend": "local", "format": "ba[protocol=https][acodec=opus]/ba[protocol=https]", "workers": 5,
                 "connections": 16, "cookies": "", "host": "", "remote_dir": "", "remote_python": "python3",
                 "stall_minutes": 30, "audio_dir": "", "pull": True},
    "diarize": {"model": "pyannote/speaker-diarization-3.1"},
    "segment": {"sample_rate": 48000, "min_speaker_minutes": 3.0, "min_dur": 3.0, "max_dur": 15.0,
                "join_gap": 0.6, "pad": 0.1},
    "screen": {"enabled": True, "batch": 64, "smooth": 0.3,
               "allow": ["Speech", "Male speech, man speaking", "Female speech, woman speaking",
                         "Child speech, kid speaking", "Conversation", "Narration, monologue",
                         "Silence", "Inside, small room"]},
    "demucs": {"enabled": False, "model": "htdemucs", "batch": 32},
    "transcribe": {"engine": "gemini", "google_language": "", "workers": 16, "model": "gemini-3.8-flash", "thinking": "low", "concurrency": 64, "audio": "segments",
                   "language": "", "code_switch": "", "orthography": "", "context": "a podcast",
                   "example": "", "prompt": ""},
    "crosscheck": {"enabled": True, "model": "omniASR_CTC_1B_v2", "python": "", "batch": 32},
    "build": {"max_nonspeech": 0.3, "max_cer": 50, "drop_digits": True, "min_cps": 4, "max_cps": 30, "rate_outlier_z": "auto", "val_episodes": 2, "seed": 0},
}


@dataclass
class Config:
    path: Path
    raw: dict

    def __getitem__(self, section):
        return self.raw[section]

    @property
    def base(self):
        return self.path.parent

    def resolve(self, p):
        p = Path(os.path.expanduser(p))
        return p if p.is_absolute() else (self.base / p).resolve()

    @property
    def data(self):
        return self.resolve(self.raw["data_dir"])

    def dir(self, name):
        d = self.data / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def audio_dir(self):
        a = self.raw["download"]["audio_dir"]
        return self.resolve(a) if a else self.dir("yt") / "videos"

    def ids(self):
        f = self.data / "videos.tsv"
        if not f.exists():
            sys.exit(f"{f} missing: run `yt-tts-dataset download` first")
        return [l.split("\t")[0] for l in f.read_text().splitlines() if l.strip()]


def load(path="config.toml"):
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    cfg = {k: ({**v, **raw.get(k, {})} if isinstance(v, dict) else raw.get(k, v)) for k, v in DEFAULTS.items()}
    env = path.parent / ".env"  # KEY=value lines (GEMINI_API_KEY, HF_TOKEN), never overriding the shell
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    return Config(path, cfg)


def write_jsonl(path, rows):
    """Atomic write: a stage's output file only exists once it is complete (that is the resume marker)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def each_episode(cfg, upstream, out, fn, follow=False, poll=60):
    """Run fn(id) for every episode whose `upstream(id)` file exists and `out(id)` does not.

    With follow=True keep polling until every episode is done, so a stage can run alongside the stage
    feeding it."""
    ids = cfg.ids()
    while True:
        todo = [v for v in ids if not Path(out(v)).exists()]
        ready = [v for v in todo if Path(upstream(v)).exists()]
        for v in ready:
            fn(v)
        if not follow or len(ready) == len(todo):
            return
        time.sleep(poll)
