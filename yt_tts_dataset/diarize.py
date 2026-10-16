"""Stage 2: mono FLAC master at segment.sample_rate + speaker diarization (RTTM) per episode.

Diarization is used only to keep every segment single-speaker and free of crosstalk; no speaker IDs
are carried into the dataset. Engines (diarize.engine):
  pyannote    (default) pyannote.audio speaker-diarization-3.1 on the GPU (HF_TOKEN, gated model). Its
              clustering and audio handling run on the CPU, so it slows down a lot on a busy machine.
  sortformer  NVIDIA Streaming Sortformer (NeMo): one end-to-end GPU model, no clustering, up to 4
              speakers; ~170x realtime on an H200 vs ~10-17x for pyannote (busy CPU), 17% DER against
              pyannote with the same main speakers. NeMo is heavy, so give it its own venv and set
              diarize.python; the stage re-runs itself there.
  sherpa      sherpa-onnx (github.com/k2-fsa/sherpa-onnx): pyannote segmentation-3.0 + WeSpeaker ResNet34
              embeddings + fast clustering, ONNX, CPU (or provider = "cuda" with its CUDA wheel). No token;
              models downloaded once. Tune cluster_threshold: the default over-splits speakers.
  none        no diarization, for sources where one person speaks throughout (a solo channel): each
              episode is one speaker turn covering the whole file, so segment cuts at VAD pauses only.
              Nothing to load, no GPU. Other voices (an intro, a clip) are not removed.
Shard across processes with --shard i --nshards n.
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .config import each_episode
from .download import audio_path

# pyannote 3.x checkpoints predate torch 2.6's weights_only default
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = f"{SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"


def cached(url, name=None):
    """Download url once into ~/.cache/yt-tts-dataset; .tar.bz2 archives are unpacked."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "yt-tts-dataset"
    cache.mkdir(parents=True, exist_ok=True)
    fname = name or url.rsplit("/", 1)[-1]
    path = cache / fname
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    if fname.endswith(".tar.bz2"):
        out = cache / fname[: -len(".tar.bz2")]
        if not out.exists():
            subprocess.run(["tar", "xjf", str(path), "-C", str(cache)], check=True)
        return out
    return path


def load16k(flac):
    """Decode to 16 kHz mono float32 with ffmpeg (one fast native pass, no Python resampling)."""
    import numpy as np

    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(flac), "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def sherpa_engine(d):
    import sherpa_onnx

    seg = cached(SEGMENTATION) / "model.onnx"
    emb = cached(d["embedding_model"])
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)),
            num_threads=d["threads"], provider=d["provider"]),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=d["threads"],
                                                              provider=d["provider"]),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=d["cluster_threshold"]),
        min_duration_on=0.3, min_duration_off=0.5)
    if not config.validate():
        raise SystemExit("sherpa-onnx diarization config invalid (models missing?)")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def diarize(flac):
        audio = load16k(flac)
        turns = [(r.start, r.end, f"SPEAKER_{r.speaker:02d}") for r in sd.process(audio).sort_by_start_time()]
        return turns, len(audio) / 16000
    return diarize


def sortformer_engine(d):
    import tempfile

    import soundfile as sf
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.from_pretrained(d["sortformer_model"], map_location="cuda").eval()
    m = model.sortformer_modules  # streaming settings from the model card (long-audio, higher-latency setup)
    m.chunk_len, m.chunk_right_context, m.fifo_len, m.spkcache_update_period, m.spkcache_len = 340, 40, 40, 300, 188
    m._check_streaming_parameters()

    def diarize(flac):
        audio = load16k(flac)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, 16000)
            for attempt in range(20):  # long episodes peak at ~10 GB; on a shared GPU wait for room
                try:
                    with torch.inference_mode():
                        segs = model.diarize(audio=[f.name], batch_size=1, verbose=False)[0]
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{flac.name}: CUDA out of memory, retrying in 60 s", flush=True)
                    time.sleep(60)
            else:
                raise RuntimeError(f"{flac}: CUDA out of memory 20 times")
        turns = []
        for s in segs:
            start, end, spk = s.split()[:3]
            turns.append((float(start), float(end), f"SPEAKER_{int(spk.rsplit('_', 1)[-1]):02d}"))
        return sorted(turns), len(audio) / 16000
    return diarize


def pyannote_engine(d):
    import torch
    from pyannote.audio import Pipeline

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pipe = Pipeline.from_pretrained(d["model"], use_auth_token=os.environ.get("HF_TOKEN"))
    pipe.to(torch.device("cuda"))

    def diarize(flac):
        audio = torch.from_numpy(load16k(flac).copy())[None]
        diar = pipe({"waveform": audio, "sample_rate": 16000})
        turns = [(t.start, t.end, lb) for t, _, lb in diar.itertracks(yield_label=True)]
        return turns, audio.shape[1] / 16000
    return diarize


def none_engine(d):
    import soundfile as sf

    def diarize(flac):
        secs = sf.info(str(flac)).duration
        return [(0.0, secs, "SPEAKER_00")], secs
    return diarize


def run(cfg, follow=False, shard=0, nshards=1):
    d = cfg["diarize"]
    py = d["python"] and cfg.executable(d["python"])
    if py and os.path.abspath(py) != os.path.abspath(sys.executable) and not os.environ.get("YTTD_CHILD"):
        env = {**os.environ, "YTTD_CHILD": "1",
               "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).parent.parent), os.environ.get("PYTHONPATH")]))}
        cmd = [py, "-m", "yt_tts_dataset.cli", "-c", str(cfg.path), "diarize", "--shard", str(shard),
               "--nshards", str(nshards)] + (["--follow"] if follow else [])
        os.execve(py, cmd, env)  # replace this process, so stopping it stops the engine too

    engines = {"sherpa": sherpa_engine, "sortformer": sortformer_engine, "pyannote": pyannote_engine,
               "none": none_engine}
    if d["engine"] not in engines:
        raise SystemExit(f"diarize.engine must be one of {sorted(engines)}")
    diarize = engines[d["engine"]](d)
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
        turns, secs = diarize(flac)
        tmp = diar_dir / f"{vid}.rttm.tmp"
        with open(tmp, "w") as f:
            f.writelines(f"SPEAKER {vid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>\n" for s, e, spk in turns)
        os.replace(tmp, diar_dir / f"{vid}.rttm")
        total = {}
        for s, e, spk in turns:
            total[spk] = total.get(spk, 0) + e - s
        main = ", ".join(f"{k}={v / 60:.1f}m" for k, v in sorted(total.items())
                         if v >= 60 * cfg["segment"]["min_speaker_minutes"])
        print(f"{vid}: {secs / 60:.0f} min in {time.time() - t0:.0f}s ({d['engine']}), main speakers {main}", flush=True)

    each_episode(cfg, lambda v: audio_path(cfg, v) or "/nonexistent", lambda v: diar_dir / f"{v}.rttm", one, follow)
