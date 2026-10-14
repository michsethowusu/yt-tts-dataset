"""Stage 2: mono FLAC master at segment.sample_rate + pyannote diarization (RTTM) per episode.

Diarization is used only to keep every segment single-speaker and free of crosstalk; no speaker IDs
are carried into the dataset. Shard across GPU processes with --shard i --nshards n.
"""
import os
import subprocess
import time

from .config import each_episode
from .download import audio_path

# pyannote 3.x checkpoints predate torch 2.6's weights_only default
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def run(cfg, follow=False, shard=0, nshards=1):
    import torch
    import torchaudio
    from pyannote.audio import Pipeline

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pipe = Pipeline.from_pretrained(cfg["diarize"]["model"], use_auth_token=os.environ.get("HF_TOKEN"))
    pipe.to(torch.device("cuda"))
    sr = cfg["segment"]["sample_rate"]
    flac_dir, diar_dir = cfg.dir("flac"), cfg.dir("diar")
    mine = set(cfg.ids()[shard::nshards])

    def one(vid):
        if vid not in mine:
            return
        flac = flac_dir / f"{vid}.flac"
        if not flac.exists():
            tmp = flac_dir / f"{vid}.tmp.flac"
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(audio_path(cfg, vid)),
                            "-ac", "1", "-ar", str(sr), "-c:a", "flac", str(tmp)], check=True)
            os.replace(tmp, flac)
        t0 = time.time()
        w, fsr = torchaudio.load(str(flac))
        w = torchaudio.functional.resample(w, fsr, 16000)
        diar = pipe({"waveform": w, "sample_rate": 16000})
        diar.uri = vid
        tmp = diar_dir / f"{vid}.rttm.tmp"
        with open(tmp, "w") as f:
            diar.write_rttm(f)
        os.replace(tmp, diar_dir / f"{vid}.rttm")
        spk = ", ".join(f"{l}={diar.label_duration(l) / 60:.1f}m" for l in diar.labels()
                        if diar.label_duration(l) >= 60 * cfg["segment"]["min_speaker_minutes"])
        print(f"{vid}: {w.shape[1] / 16000 / 60:.0f} min in {time.time() - t0:.0f}s, main speakers {spk}", flush=True)

    each_episode(cfg, lambda v: audio_path(cfg, v) or "/nonexistent", lambda v: diar_dir / f"{v}.rttm", one, follow)
