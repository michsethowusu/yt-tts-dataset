"""Stage 6: omniASR CTC second opinion on every transcribed segment.

An LLM transcriber can hallucinate fluent text on music, laughter or unclear audio. A CTC model can't
invent words that aren't in the audio, so a wide disagreement between the two flags a bad transcript.
xcheck/<id>.jsonl: text rows plus
  omni      the CTC transcript (no language tag; CTC handles code-switching on its own)
  cer       CER(text -> omni) in %, on lowercase letters only with spaces removed, so casing,
            punctuation and CTC's merged words are not counted as disagreement
  digits    the transcript contains digits
omnilingual-asr pins its own torch/fairseq2, so it usually lives in its own venv: set crosscheck.python
and this stage re-runs itself there (the package is put on PYTHONPATH, no install needed).
"""
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

from .config import each_episode, read_jsonl, write_jsonl


def letters(t):
    t = unicodedata.normalize("NFC", (t or "").lower())
    return "".join(c for c in t if unicodedata.category(c)[0] == "L")


def cer(ref, hyp):
    a, b = letters(ref), letters(hyp)
    if not a:
        return 100.0
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return 100 * prev[-1] / len(a)


def run(cfg, follow=False):
    c = cfg["crosscheck"]
    py = c["python"] and str(cfg.resolve(c["python"]))
    # compare unresolved: venv pythons are symlinks to the same interpreter
    if py and os.path.abspath(py) != os.path.abspath(sys.executable) and not os.environ.get("YTTD_CHILD"):
        env = {**os.environ, "YTTD_CHILD": "1",
               "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).parent.parent), os.environ.get("PYTHONPATH")]))}
        cmd = [py, "-m", "yt_tts_dataset.cli", "-c", str(cfg.path), "crosscheck"] + (["--follow"] if follow else [])
        os.execve(py, cmd, env)  # replace this process, so stopping it stops the omni run too

    import torch  # noqa: F401  (load before fairseq2's audio extension)
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    pipe = ASRInferencePipeline(model_card=c["model"])
    text_dir, out_dir = cfg.dir("text"), cfg.dir("xcheck")

    def one(vid):
        t0 = time.time()
        rows = [r for r in read_jsonl(text_dir / f"{vid}.jsonl") if r["text"]]
        hyps = pipe.transcribe([r["audio"] for r in rows], batch_size=c["batch"]) if rows else []
        for r, h in zip(rows, hyps):
            r.update(omni=h, cer=round(cer(r["text"], h), 1), digits=bool(re.search(r"\d", r["text"])))
        write_jsonl(out_dir / f"{vid}.jsonl", rows)
        cs = sorted(r["cer"] for r in rows) or [0]
        print(f"{vid}: {len(rows)} segments, CER median {cs[len(cs) // 2]:.0f}% p90 {cs[int(len(cs) * .9)]:.0f}% "
              f"in {time.time() - t0:.0f}s", flush=True)

    each_episode(cfg, lambda v: text_dir / f"{v}.jsonl", lambda v: out_dir / f"{v}.jsonl", one, follow)
