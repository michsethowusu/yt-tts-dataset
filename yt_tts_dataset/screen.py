"""Stage 4: screen every segment for sound that isn't plain speech (music, laughter, applause, singing, noise).

PANNs CNN14 sound event detection (AudioSet's 527 classes, 10 ms frames, github.com/qiuqiangkong/
panns_inference) on the GPU. Each class's frame scores are averaged over screen.smooth seconds, so a
short laugh still counts but a single-frame blip does not; a segment's `nonspeech` score is the
highest smoothed score of any class outside screen.allow (speech, silence, a small room).
screen/<id>.jsonl: audio plus
  nonspeech        0-1; `build` drops segments above build.max_nonspeech (see `report` first)
  nonspeech_label  the class that scored it, e.g. "Laughter", "Music"
  speech           highest smoothed score of the "Speech" class
Much faster than Demucs, and drops the segment instead of trying to clean it.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from .config import each_episode, read_jsonl, write_jsonl

SR = 32000  # PANNs input rate
HOP = 320   # 10 ms frames


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
    import torch.nn.functional as F
    import torchaudio
    from panns_inference import SoundEventDetection, labels

    c = cfg["screen"]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = SoundEventDetection(checkpoint_path=None, device="cuda")
    allow = [i for i, lb in enumerate(labels) if lb in set(c["allow"])]
    unknown = set(c["allow"]) - {labels[i] for i in allow}
    if unknown:
        raise SystemExit(f"screen.allow: not AudioSet class names: {sorted(unknown)}")
    other = torch.ones(len(labels), dtype=torch.bool)
    other[allow] = False
    speech = labels.index("Speech")
    win = max(1, round(c["smooth"] * SR / HOP))
    seg_dir, out_dir = cfg.dir("segs"), cfg.dir("screen")
    resamplers = {}

    def load(path):
        w, sr = sf.read(path, dtype="float32", always_2d=True)
        w = torch.from_numpy(w.mean(1)).cuda()
        if sr != SR:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, SR).cuda()
            w = resamplers[sr](w)
        return w

    def one(vid):
        t0 = time.time()
        rows = read_jsonl(seg_dir / f"{vid}.jsonl")
        res = {}
        with ThreadPoolExecutor(8) as pool:
            for batch in batches_by_length(rows, c["batch"], c["batch"] * 8):
                wavs = list(pool.map(load, [r["audio"] for r in batch]))
                n = max(len(w) for w in wavs)
                x = torch.stack([F.pad(w, (0, n - len(w))) for w in wavs])
                fw = torch.from_numpy(model.inference(x))  # (B, frames, 527)
                fw = F.avg_pool1d(fw.transpose(1, 2), win, stride=1, padding=win // 2, count_include_pad=False).transpose(1, 2)
                for r, w, f in zip(batch, wavs, fw):
                    f = f[: max(1, -(-len(w) // HOP))]  # drop padding frames
                    peak = f.max(0).values
                    o = peak.masked_fill(~other, -1)
                    k = int(o.argmax())
                    res[r["audio"]] = {"audio": r["audio"], "nonspeech": round(float(o[k]), 3),
                                       "nonspeech_label": labels[k], "speech": round(float(peak[speech]), 3)}
        out = [res[r["audio"]] for r in rows]
        write_jsonl(out_dir / f"{vid}.jsonl", out)
        s = np.array([r["nonspeech"] for r in out] or [0])
        print(f"{vid}: {len(out)} segments, nonspeech median {np.median(s):.2f}, >0.3 {np.mean(s > 0.3):.0%} "
              f"in {time.time() - t0:.0f}s", flush=True)

    each_episode(cfg, lambda v: seg_dir / f"{v}.jsonl", lambda v: out_dir / f"{v}.jsonl", one, follow)
