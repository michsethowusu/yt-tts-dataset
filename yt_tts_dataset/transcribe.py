"""Stage 5: Gemini transcription of every segment (async, many requests in flight).

text/<id>.jsonl: the segment rows plus
  text      verbatim transcript with Gemini's casing and punctuation (TTS frameworks normalise);
            "" = no intelligible speech, null = API failed after all retries (rerun the stage to retry)
  in_tok, out_tok   token usage (out includes thinking)
Each episode is written once all its segments are back; the stage is resume-safe per episode.
"""
import asyncio
import base64
import os
import random
import time
from pathlib import Path

from .config import read_jsonl, write_jsonl

TEMPLATE = (
    "This is a clip from {context}. The speaker talks {language}{switch}. Transcribe exactly what is said, "
    "word for word, for a text-to-speech dataset:\n"
    "- {language} in {orthography}{switch_spelling}. Do not translate anything.\n"
    "- Write numbers as words, in the language they were spoken in.\n"
    "- Keep fillers and repetitions as spoken. No speaker labels, no timestamps, no notes.\n"
    "- If a word is cut off at the very start or end of the clip, leave it out.\n"
    "Put the transcription inside square brackets{example}. If there is no intelligible speech, output []."
)


def build_prompt(t):
    if t["prompt"]:
        return t["prompt"]
    cs = t["code_switch"]
    return TEMPLATE.format(
        context=t["context"], language=t["language"] or "its language",
        switch=f" and often code-switches into {cs} mid-sentence" if cs else "",
        orthography=t["orthography"] or "its standard orthography",
        switch_spelling=f", {cs} words in normal {cs} spelling" if cs else "",
        example=f", e.g. [{t['example']}]" if t["example"] else "")


async def gemini(session, sem, path, t, prompt, key):
    import aiohttp

    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "audio/flac", "data": base64.b64encode(Path(path).read_bytes()).decode()}},
        {"text": prompt}]}],
        "generationConfig": {"temperature": 0, "thinkingConfig": {"thinkingLevel": t["thinking"]}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{t['model']}:generateContent"
    for attempt in range(8):
        async with sem:
            try:
                async with session.post(url, json=body, headers={"x-goog-api-key": key},
                                        timeout=aiohttp.ClientTimeout(total=180)) as resp:
                    if resp.status == 200:
                        r = await resp.json()
                        u = r.get("usageMetadata", {})
                        parts = (r.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                        raw = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                        if raw:
                            s, e = raw.find("["), raw.rfind("]")
                            return (raw[s + 1:e].strip() if s >= 0 and e > s else raw, u.get("promptTokenCount", 0),
                                    u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0))
                    elif resp.status not in (429, 500, 502, 503, 504):
                        print(f"{path}: HTTP {resp.status} {(await resp.text())[:200]}", flush=True)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
        await asyncio.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
    return None, 0, 0


async def _run(cfg, follow):
    import aiohttp

    t = cfg["transcribe"]
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set (shell or .env next to the config)")
    prompt = build_prompt(t)
    use_vocals = t["audio"] == "vocals"
    src_dir = cfg.dir("vocals" if use_vocals else "segs")
    seg_dir, out_dir = cfg.dir("segs"), cfg.dir("text")
    ids = cfg.ids()
    sem = asyncio.Semaphore(t["concurrency"])

    async def one(session, vid):
        t0 = time.time()
        rows = read_jsonl(seg_dir / f"{vid}.jsonl")
        if use_vocals:
            voc = {r["audio"]: r["vocals"] for r in read_jsonl(src_dir / f"{vid}.jsonl")}
        res = await asyncio.gather(*(gemini(session, sem, voc[r["audio"]] if use_vocals else r["audio"], t, prompt, key)
                                     for r in rows))
        out = [{**r, "text": text, "in_tok": tin, "out_tok": tout} for r, (text, tin, tout) in zip(rows, res)]
        write_jsonl(out_dir / f"{vid}.jsonl", out)
        print(f"{vid}: {len(rows)} segments, empty {sum(x[0] == '' for x in res)}, failed {sum(x[0] is None for x in res)}, "
              f"tokens {sum(x[1] for x in res)}/{sum(x[2] for x in res)} in {time.time() - t0:.0f}s", flush=True)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=t["concurrency"])) as session:
        running = {}
        while True:
            todo = [v for v in ids if not (out_dir / f"{v}.jsonl").exists()]
            for v in todo:
                if v not in running and (src_dir / f"{v}.jsonl").exists():
                    running[v] = asyncio.create_task(one(session, v))
            for v in [v for v, task in running.items() if task.done()]:
                running.pop(v).result()
            if not todo or (not follow and not running):
                break
            await asyncio.sleep(10)


def run(cfg, follow=False):
    asyncio.run(_run(cfg, follow))


def retry_failed(cfg):
    """Re-request segments that came back null, in place."""
    async def go():
        import aiohttp

        t, key, prompt = cfg["transcribe"], os.environ["GEMINI_API_KEY"], build_prompt(cfg["transcribe"])
        sem = asyncio.Semaphore(t["concurrency"])
        async with aiohttp.ClientSession() as session:
            for f in sorted(cfg.dir("text").glob("*.jsonl")):
                rows = read_jsonl(f)
                bad = [r for r in rows if r["text"] is None]
                if not bad:
                    continue
                paths = [str(cfg.dir("vocals") / r["episode"] / Path(r["audio"]).name) if t["audio"] == "vocals"
                         else r["audio"] for r in bad]
                res = await asyncio.gather(*(gemini(session, sem, p, t, prompt, key) for p in paths))
                for r, (text, tin, tout) in zip(bad, res):
                    r.update(text=text, in_tok=tin, out_tok=tout)
                write_jsonl(f, rows)
                print(f"{f.stem}: retried {len(bad)}, still failed {sum(x[0] is None for x in res)}", flush=True)
    asyncio.run(go())
