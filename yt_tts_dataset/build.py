"""Stage 7: filter and write the dataset.

`report` prints how many hours each filter setting keeps (look before you drop);
`build` applies [build] (non-speech score, CER, digits, characters/second, speaking-rate outliers) and
writes dataset/{train,val}.jsonl plus LJSpeech-style metadata_{train,val}.csv (audio|text). Audio is the
Demucs vocals when demucs.enabled. Validation = whole held-out episodes.

Speaking-rate outliers: an ASR that drops words (Google's free endpoint truncates long or unclear clips)
leaves a transcript far too short for its audio, one that runs words together or hallucinates leaves one
too long. Rate = letters per second (no spaces/punctuation, so Google's bare text and Gemini's punctuated
text compare alike); rate_z = robust z-score of log(rate) against the whole dataset (median, 1.4826 x MAD).
Segments with |rate_z| > build.rate_outlier_z are dropped; "auto" = 3 for the google engine, off for gemini.

Digits: their spoken form is ambiguous, so drop_digits drops transcripts containing any. build.spell_digits =
a num2words language code (e.g. "en") first writes them out as words: years ("1979", "1980s"), ordinals ("2nd"),
percentages, $/£/€ amounts, decimals and plain numbers ("1,000"); anything left over is still dropped.
Years read as years: 1979 = "nineteen seventy-nine", 2021 = "twenty twenty-one" (2000-2009 = "two thousand
and nine"); "and" after hundreds ("one hundred and five"). Use "en" for numbers inside other languages too: an
ASR writes digits for numbers said in English (one said in Twi comes back as Twi words).
"""
import math
import random
import re

from .config import read_jsonl, write_jsonl
from .crosscheck import letters

NUMBER = re.compile(r"([$£€])?(\d[\d,]*(?:\.\d+)?)(st|nd|rd|th|s|%)?(?![\w])", re.I)
CURRENCY = {"$": ("dollar", "dollars", "USD"), "£": ("pound", "pounds", "GBP"), "€": ("euro", "euros", "EUR")}


def spell_digits(text, lang):
    """Write the numbers in text out as words with num2words; the year/ordinal/plural reading comes from the form."""
    from num2words import num2words

    def word(m):
        cur, num, suffix = m.group(1), m.group(2).replace(",", ""), (m.group(3) or "").lower()
        try:
            if cur and "." not in num:  # $5 -> five dollars (not "five dollars, zero cents")
                w = num2words(int(num), lang=lang) + " " + CURRENCY[cur][int(num) != 1]
            elif cur:
                w = num2words(float(num), lang=lang, to="currency", currency=CURRENCY[cur][2])
            elif suffix in ("st", "nd", "rd", "th"):
                w = num2words(int(num), lang=lang, to="ordinal")
            elif "." not in num and len(num) == 4 and 1100 <= int(num) <= 2099:
                w = num2words(int(num), lang=lang, to="year")
                if suffix == "s":  # 1980s -> nineteen eighties
                    w = w[:-1] + "ies" if w.endswith("y") else w + "s"
            else:
                w = num2words(float(num) if "." in num else int(num), lang=lang)
                w += {"%": " percent", "s": "s"}.get(suffix, "")
        except (ValueError, NotImplementedError, OverflowError):
            return m.group(0)
        return w.replace(",", "")
    return NUMBER.sub(word, text)


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
    lang = cfg["build"]["spell_digits"]
    for r in rows:
        if lang and re.search(r"\d", r["text"]):
            r["text"] = spell_digits(r["text"], lang)
        r["cps"] = round(len(r["text"]) / r["dur"], 2)
        r["rate"] = len(letters(r["text"])) / r["dur"]
        r["digits"] = bool(re.search(r"\d", r["text"]))  # spoken form is ambiguous ("2" = two / to / second)
    logs = sorted(math.log(r["rate"]) for r in rows if r["rate"] > 0)
    if logs:
        med = logs[len(logs) // 2]
        mad = 1.4826 * sorted(abs(x - med) for x in logs)[len(logs) // 2] or 1.0
        for r in rows:
            r["rate_z"] = round((math.log(r["rate"]) - med) / mad, 2) if r["rate"] > 0 else -99.0
    return rows


def rate_z_limit(cfg):
    z = cfg["build"]["rate_outlier_z"]
    if z == "auto":
        return 3.0 if cfg["transcribe"]["engine"] == "google" else 0
    return float(z or 0)


def hours(rows):
    return sum(r["dur"] for r in rows) / 3600


def keep(r, b, max_cer=None, max_nonspeech=None, rate_z=None):
    max_cer = b["max_cer"] if max_cer is None else max_cer
    max_ns = b["max_nonspeech"] if max_nonspeech is None else max_nonspeech
    rate_z = b["_rate_z"] if rate_z is None else rate_z
    return ((r.get("cer") is None or r["cer"] <= max_cer) and (r.get("nonspeech") is None or r["nonspeech"] <= max_ns)
            and not (b["drop_digits"] and r.get("digits")) and b["min_cps"] <= r["cps"] <= b["max_cps"]
            and (not rate_z or abs(r.get("rate_z", 0)) <= rate_z))


def report(cfg):
    b = {**cfg["build"], "_rate_z": rate_z_limit(cfg)}
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
    if rows:
        rs = sorted(r["rate"] for r in rows)
        print(f"\nspeaking rate (letters/s): median {rs[len(rs) // 2]:.1f}, 1-99% {rs[len(rs) // 100]:.1f}-{rs[len(rs) * 99 // 100]:.1f}")
        print("rate_outlier_z   too slow  too fast   kept (other filters applied)")
        for z in (2.0, 2.5, 3.0, 3.5, 4.0, 0):
            lo = [r for r in rows if z and r["rate_z"] < -z]
            hi = [r for r in rows if z and r["rate_z"] > z]
            k = [r for r in rows if keep(r, b, rate_z=z)]
            print(f"  {z if z else 'off':>4}   {hours(lo):6.1f} h  {hours(hi):6.1f} h   {len(k):6d} segments  {hours(k):6.1f} h"
                  + ("   <- config" if z == b["_rate_z"] else ""))
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
    b = {**cfg["build"], "_rate_z": rate_z_limit(cfg)}
    rows = [r for r in load_rows(cfg) if keep(r, b)]
    eps = sorted({r["episode"] for r in rows})
    val_eps = set(random.Random(b["seed"]).sample(eps, min(b["val_episodes"], max(0, len(eps) - 1))))
    out = cfg.dir("dataset")
    fields = ("audio", "text", "dur", "episode", "start", "end", "rate_z", "nonspeech", "nonspeech_label", "cer",
              "omni", "removed_db", "source_audio")
    for split in ("train", "val"):
        part = [r for r in rows if (r["episode"] in val_eps) == (split == "val")]
        part = [{k: v for k, v in {**r, "source_audio": r["audio"], "audio": r.get("vocals", r["audio"])}.items()
                 if k in fields} for r in part]
        write_jsonl(out / f"{split}.jsonl", part)
        with open(out / f"metadata_{split}.csv", "w") as f:
            f.writelines(f"{r['audio']}|{' '.join(r['text'].split())}\n" for r in part)
        print(f"{split}: {len(part)} segments, {hours(part):.1f} h, {len({r['episode'] for r in part})} episodes")
    print(f"-> {out}")
