"""Stage 7: filter and write the dataset.

`report` prints how many hours each filter setting keeps (look before you drop);
`build` applies [build] (non-speech score, CER, digits, characters/second) and writes dataset/{train,val}.jsonl plus LJSpeech-style metadata_{train,val}.csv
(audio|text). Audio is the Demucs vocals when demucs.enabled. Validation = whole held-out episodes.
"""
import random

from .config import read_jsonl, write_jsonl


def load_rows(cfg):
    use_x = cfg["crosscheck"]["enabled"]
    src = cfg.dir("xcheck" if use_x else "text")
    rows = [r for f in sorted(src.glob("*.jsonl")) for r in read_jsonl(f) if r.get("text")]
    if cfg["demucs"]["enabled"]:
        voc = {r["audio"]: r for f in sorted(cfg.dir("vocals").glob("*.jsonl")) for r in read_jsonl(f)}
        rows = [{**r, "vocals": voc[r["audio"]]["vocals"], "removed_db": voc[r["audio"]]["removed_db"]}
                for r in rows if r["audio"] in voc]
    if cfg["screen"]["enabled"]:
        scr = {r["audio"]: r for f in sorted(cfg.dir("screen").glob("*.jsonl")) for r in read_jsonl(f)}
        rows = [{**scr[r["audio"]], **r} for r in rows if r["audio"] in scr]
    for r in rows:
        r["cps"] = round(len(r["text"]) / r["dur"], 2)
    return rows


def hours(rows):
    return sum(r["dur"] for r in rows) / 3600


def keep(r, b, max_cer=None, max_nonspeech=None):
    max_cer = b["max_cer"] if max_cer is None else max_cer
    max_ns = b["max_nonspeech"] if max_nonspeech is None else max_nonspeech
    return ((r.get("cer") is None or r["cer"] <= max_cer) and (r.get("nonspeech") is None or r["nonspeech"] <= max_ns) and not (b["drop_digits"] and r.get("digits"))
            and b["min_cps"] <= r["cps"] <= b["max_cps"])


def report(cfg):
    b = cfg["build"]
    seg = [r for f in sorted(cfg.dir("segs").glob("*.jsonl")) for r in read_jsonl(f)]
    txt = [r for f in sorted(cfg.dir("text").glob("*.jsonl")) for r in read_jsonl(f)]
    rows = load_rows(cfg)
    print(f"segments           {len(seg):6d}  {hours(seg):6.1f} h  ({len({r['episode'] for r in seg})} episodes)")
    print(f"transcribed        {sum(1 for r in txt if r['text']):6d}  {hours([r for r in txt if r['text']]):6.1f} h"
          f"  (empty {sum(r['text'] == '' for r in txt)}, failed {sum(r['text'] is None for r in txt)})")
    tin, tout = sum(r.get("in_tok", 0) for r in txt), sum(r.get("out_tok", 0) for r in txt)
    if cfg["transcribe"]["engine"] == "gemini":
        print(f"gemini tokens      in {tin / 1e6:.2f}M  out+thinking {tout / 1e6:.2f}M")
    print(f"with digits        {sum(1 for r in rows if r.get('digits')):6d}  {hours([r for r in rows if r.get('digits')]):6.1f} h")
    out = [r for r in rows if not b["min_cps"] <= r["cps"] <= b["max_cps"]]
    print(f"cps outside {b['min_cps']}-{b['max_cps']}  {len(out):6d}  {hours(out):6.1f} h")
    if cfg["screen"]["enabled"]:
        from collections import Counter
        print("\nmax_nonspeech   kept (other filters applied)")
        for m in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            k = [r for r in rows if keep(r, b, max_nonspeech=m)]
            print(f"  {m:4.1f}   {len(k):6d} segments  {hours(k):6.1f} h" + ("   <- config" if m == b["max_nonspeech"] else ""))
        drop = Counter(r["nonspeech_label"] for r in rows if r["nonspeech"] > b["max_nonspeech"])
        print("  dropped at config, by sound: " + ", ".join(f"{k} {v}" for k, v in drop.most_common(12)))
    if cfg["crosscheck"]["enabled"]:
        print("\nmax_cer   kept (digits/cps filters applied)")
        for m in (20, 30, 40, 50, 60, 80, 100):
            k = [r for r in rows if keep(r, b, m)]
            print(f"  {m:4d}%   {len(k):6d} segments  {hours(k):6.1f} h" + ("   <- config" if m == b["max_cer"] else ""))


def run(cfg):
    b = cfg["build"]
    rows = [r for r in load_rows(cfg) if keep(r, b)]
    eps = sorted({r["episode"] for r in rows})
    val_eps = set(random.Random(b["seed"]).sample(eps, min(b["val_episodes"], max(0, len(eps) - 1))))
    out = cfg.dir("dataset")
    fields = ("audio", "text", "dur", "episode", "start", "end", "nonspeech", "nonspeech_label", "cer", "omni",
              "removed_db", "source_audio")
    for split in ("train", "val"):
        part = [r for r in rows if (r["episode"] in val_eps) == (split == "val")]
        part = [{k: v for k, v in {**r, "source_audio": r["audio"], "audio": r.get("vocals", r["audio"])}.items()
                 if k in fields} for r in part]
        write_jsonl(out / f"{split}.jsonl", part)
        with open(out / f"metadata_{split}.csv", "w") as f:
            f.writelines(f"{r['audio']}|{' '.join(r['text'].split())}\n" for r in part)
        print(f"{split}: {len(part)} segments, {hours(part):.1f} h, {len({r['episode'] for r in part})} episodes")
    print(f"-> {out}")
