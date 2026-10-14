"""Stage 4: Demucs vocals stem for every segment (removes music beds, jingles, background noise).

segs/<id>/<seg>.flac -> 44.1 kHz stereo -> htdemucs -> vocals -> back to segment.sample_rate, mono 16-bit
vocals/<id>/<seg>.flac. vocals/<id>.jsonl: audio, vocals, removed_db (input/output RMS in dB, i.e. how
much non-vocal energy was removed), peak_in.
"""
import os
import time

from .config import each_episode, read_jsonl, write_jsonl

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # variable-length batches


def batches_by_length(rows, max_batch, max_total_s):
    batch = []
    for r in sorted(rows, key=lambda r: r["dur"]):
        if batch and (len(batch) >= max_batch or (len(batch) + 1) * r["dur"] > max_total_s):
            yield batch
            batch = []
        batch.append(r)
    if batch:
        yield batch


def run(cfg, follow=False):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = get_model(cfg["demucs"]["model"]).to(dev).eval()
    voc_idx = model.sources.index("vocals")
    sr = cfg["segment"]["sample_rate"]
    up = torchaudio.transforms.Resample(sr, model.samplerate).to(dev)
    down = torchaudio.transforms.Resample(model.samplerate, sr).to(dev)
    bs = cfg["demucs"]["batch"]
    seg_dir, voc_dir = cfg.dir("segs"), cfg.dir("vocals")

    @torch.no_grad()
    def one(vid):
        t0 = time.time()
        (voc_dir / vid).mkdir(exist_ok=True)
        out = []
        for batch in batches_by_length(read_jsonl(seg_dir / f"{vid}.jsonl"), bs, bs * 15.0):
            wavs = [torch.from_numpy(sf.read(r["audio"], dtype="float32")[0]) for r in batch]
            x = torch.zeros(len(wavs), max(w.shape[0] for w in wavs))
            for i, w in enumerate(wavs):
                x[i, : w.shape[0]] = w
            x = up(x.to(dev))
            x = torch.stack([x, x], dim=1)  # (B, 2, T): htdemucs expects stereo
            ref = x.mean(1, keepdim=True)
            mean, std = ref.mean(-1, keepdim=True), ref.std(-1, keepdim=True) + 1e-8
            y = apply_model(model, (x - mean) / std, device=dev, shifts=0, split=True, overlap=0.25, progress=False)
            voc = down((y[:, voc_idx] * std + mean).mean(1)).cpu().numpy()
            for r, w, v in zip(batch, wavs, voc):
                w, v = w.numpy(), v[: w.shape[0]]
                path = voc_dir / vid / r["audio"].rsplit("/", 1)[-1]
                sf.write(str(path), np.clip(v, -1, 1), sr, format="FLAC", subtype="PCM_16")
                rin, rout = np.sqrt((w ** 2).mean()) + 1e-9, np.sqrt((v ** 2).mean()) + 1e-9
                out.append({"audio": r["audio"], "vocals": str(path),
                            "removed_db": round(float(20 * np.log10(rin / rout)), 2),
                            "peak_in": round(float(np.abs(w).max()), 4)})
        write_jsonl(voc_dir / f"{vid}.jsonl", sorted(out, key=lambda r: r["audio"]))
        rd = sorted(r["removed_db"] for r in out) or [0]
        print(f"{vid}: {len(out)} segments, removed_db median {rd[len(rd) // 2]:.1f} p90 {rd[int(len(rd) * .9)]:.1f} "
              f"in {time.time() - t0:.0f}s", flush=True)

    each_episode(cfg, lambda v: seg_dir / f"{v}.jsonl", lambda v: voc_dir / f"{v}.jsonl", one, follow)
