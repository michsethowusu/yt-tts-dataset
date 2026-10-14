"""Stage 3: diarized episode -> single-speaker segments cut at real pauses.

  1. main speakers = labels with >= min_speaker_minutes (drops pyannote's tiny spurious clusters)
  2. exclusive regions = a main speaker's turns minus every other label's turns (no overlap/crosstalk)
  3. Silero VAD speech inside those regions; one speaker's pieces are joined across pauses < join_gap
     (only if nobody else speaks in the pause) up to max_dur, so every cut lands in a pause
  4. min_dur..max_dur segments (+pad each side) written as mono 16-bit FLAC: segs/<id>/<id>_<n>.flac
Writes segs/<id>.jsonl: audio, episode, start, end, dur.
"""
import time

from .config import each_episode, write_jsonl


def subtract(a, b):
    """intervals a minus intervals b (both sorted lists of (s, e))."""
    out = []
    for s, e in a:
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
        if cur < e:
            out.append((cur, e))
    return out


def spans(rttm, speech, c):
    turns = {}
    for r in (l.split() for l in open(rttm) if l.strip()):
        turns.setdefault(r[7], []).append((float(r[3]), float(r[3]) + float(r[4])))
    main = [k for k, v in turns.items() if sum(e - s for s, e in v) >= c["min_speaker_minutes"] * 60]
    segs = []
    for spk in main:
        others = sorted(t for k, v in turns.items() if k != spk for t in v)
        pieces = [(max(s, vs), min(e, ve)) for s, e in subtract(sorted(turns[spk]), others)
                  for vs, ve in speech if ve > s and vs < e]
        cur = None
        for s, e in pieces:
            if (cur and s - cur[1] < c["join_gap"] and e - cur[0] <= c["max_dur"]
                    and not any(oe > cur[1] and os_ < s for os_, oe in others)):
                cur = (cur[0], e)
            else:
                if cur:
                    segs.append(cur)
                cur = (s, e)
        if cur:
            segs.append(cur)
    p = c["pad"]
    return sorted((s, e) for s, e in segs if c["min_dur"] <= e - s + 2 * p <= c["max_dur"] + 2 * p)


def run(cfg, follow=False):
    import soundfile as sf
    import torch
    import torchaudio
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad = load_silero_vad()
    c = cfg["segment"]
    flac_dir, diar_dir, seg_dir = cfg.dir("flac"), cfg.dir("diar"), cfg.dir("segs")

    def one(vid):
        t0 = time.time()
        a, sr = sf.read(str(flac_dir / f"{vid}.flac"), dtype="float32")
        a16 = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000)
        speech = [(t["start"] / 16000, t["end"] / 16000) for t in
                  get_speech_timestamps(a16, vad, sampling_rate=16000, min_silence_duration_ms=150, speech_pad_ms=0)]
        total = len(a) / sr
        out = seg_dir / vid
        out.mkdir(exist_ok=True)
        rows = []
        for n, (s, e) in enumerate(spans(diar_dir / f"{vid}.rttm", speech, c)):
            st, en = max(0.0, s - c["pad"]), min(total, e + c["pad"])
            path = out / f"{vid}_{n:04d}.flac"
            sf.write(str(path), a[int(st * sr): int(en * sr)], sr, format="FLAC", subtype="PCM_16")
            rows.append({"audio": str(path), "episode": vid, "start": round(st, 3), "end": round(en, 3),
                         "dur": round(en - st, 3)})
        write_jsonl(seg_dir / f"{vid}.jsonl", rows)
        print(f"{vid}: {len(rows)} segments, {sum(r['dur'] for r in rows) / 3600:.2f} h in {time.time() - t0:.0f}s",
              flush=True)

    each_episode(cfg, lambda v: diar_dir / f"{v}.rttm", lambda v: seg_dir / f"{v}.jsonl", one, follow)
