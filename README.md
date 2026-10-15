# yt-tts-dataset

Turn a YouTube playlist (or any list of video URLs) into a text-to-speech dataset: full-band,
single-speaker segments with verbatim transcripts, built for conversational and code-switched speech.

```
playlist / URLs ──► download ──► diarize ──► segment ──► demucs ──► transcribe ──► crosscheck ──► build
                    yt-split-    pyannote    single-     vocals     Gemini         omniASR CTC     filter +
                    downloader   3.1         speaker     stem       (async)        second opinion  train/val
                                             3–15 s
```

Built at [Ghana NLP](https://ghananlp.org) for Twi/English podcasts, but the language, prompt and
thresholds are all in the config.

## What each stage does

| Stage | Output (`data/…`) | Why |
|---|---|---|
| **download** | `videos.tsv`, `yt/videos/<id>.mp4` | Expands playlists, drops videos outside `min_minutes`/`max_minutes` (e.g. short excerpts of full episodes), downloads best audio only (opus, 48 kHz) with [yt-split-downloader](https://github.com/michsethowusu/yt-split-downloader). |
| **diarize** | `flac/<id>.flac`, `diar/<id>.rttm` | 48 kHz mono FLAC master; pyannote 3.1 speaker turns. Used **only** to keep segments single-speaker — no speaker IDs end up in the dataset. |
| **segment** | `segs/<id>/*.flac`, `segs/<id>.jsonl` | Keeps each speaker's exclusive regions (other speakers' turns subtracted, so no overlap or crosstalk), cuts at Silero VAD pauses into 3–15 s segments. Pieces of one speaker are joined across short pauses only if nobody else speaks in the pause. |
| **demucs** | `vocals/<id>/*.flac` | htdemucs vocals stem: removes music beds, jingles and background noise. Back to the master sample rate. |
| **transcribe** | `text/<id>.jsonl` | `engine = "gemini"`: many requests in flight; verbatim, keeps casing and punctuation (leave normalisation to your TTS framework), numbers written as words. `engine = "google"`: Google's free speech endpoint in one language (`google_language`, e.g. `en-GH`), no key needed, no punctuation. |
| **crosscheck** | `xcheck/<id>.jsonl` | omniASR CTC-1B transcribes the same audio. An LLM can hallucinate fluent text on music or unclear audio; a CTC model can't invent words that aren't there. CER between the two (letters only) flags bad transcripts. |
| **build** | `dataset/{train,val}.jsonl`, `metadata_{train,val}.csv` | Drops segments above `max_cer`, with digits, or with implausible characters/second; holds out whole episodes for validation. |

Every stage is **resume-safe**: an episode's output file is written atomically when the episode is
done, and each stage skips episodes that already have one. Kill anything, rerun, nothing is redone.

## Install

Needs a CUDA GPU, `ffmpeg`/`ffprobe`, `aria2c` (downloads) and Python ≥ 3.10.

```bash
git clone https://github.com/michsethowusu/yt-tts-dataset && cd yt-tts-dataset
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

The crosscheck model ([omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)) pins
its own torch/fairseq2, so give it a separate venv and point `crosscheck.python` at it — the stage
re-runs itself there:

```bash
python -m venv .venv-omni && .venv-omni/bin/pip install omnilingual-asr
```

Keys go in the shell or a `.env` next to your config:

```
GEMINI_API_KEY=...  # only for transcribe.engine = "gemini"
HF_TOKEN=...        # accept the terms of pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 first
```

## Run

```bash
cp config.example.toml config.toml     # set source.urls, language, paths
yt-tts-dataset run                     # download, then all stages at once (each follows the previous)
yt-tts-dataset report                  # hours kept at each CER cutoff — look before you drop
yt-tts-dataset build                   # apply [build] filters, write data/dataset/
```

`run` starts every stage as its own process (logs in `data/logs/`); each picks up episodes as the
stage before it finishes them, so the GPU stages and the Gemini requests overlap. `run --sequential`
does one stage at a time. Stages can also be run alone, e.g. `yt-tts-dataset transcribe --follow`,
`yt-tts-dataset diarize --shard 0 --nshards 3` (several diarization processes on one GPU), or
`yt-tts-dataset transcribe --retry-failed`.

### Downloading

Downloads go through [yt-split-downloader](https://github.com/michsethowusu/yt-split-downloader):
media URLs are resolved on the machine running `download` and the bytes are fetched where
`download.backend` says.

- `local` — everything on one machine.
- `ssh` — for a GPU server whose datacenter IP YouTube blocks: run `download` on your own PC with
  `host` / `remote_dir` set; the server downloads the files itself. Copy `config.toml` and
  `data/videos.tsv` to the server, set `download.audio_dir = "<remote_dir>/videos"` there, and run
  the other stages on the server (`yt-tts-dataset run --skip-download`, or stages with `--follow` so
  they start while downloads are still arriving).
- `modal` — download on [Modal](https://modal.com), fetched back here.

If some videos fail, `yt-tts-dataset download --skip-expand` retries just those.

## Output

`data/dataset/train.jsonl` (and `val.jsonl`), one segment per line:

```json
{"audio": "data/vocals/<id>/<id>_0042.flac", "text": "Yeah, nti wohwɛ a, na ɛnyɛ production no a yɛreyi wɔ Kumase no",
 "dur": 3.9, "episode": "<id>", "start": 812.31, "end": 816.21, "cer": 22.9, "omni": "…", "removed_db": 0.4,
 "source_audio": "data/segs/<id>/<id>_0042.flac"}
```

`metadata_train.csv` has `audio|text` lines for frameworks that take LJSpeech-style lists.

## Notes

- **Cost.** Gemini 3.8 Flash with low thinking used ≈ 200k input + 30k output tokens per hour of audio.
- **Google engine.** Uses the unofficial free endpoint behind SpeechRecognition's `recognize_google`:
  no key or cost, but one language per run, no punctuation, and it can rate-limit you (failed requests
  are kept as `null`; `transcribe --retry-failed` re-requests them). Lower `workers` if many fail.
- **Prompt.** Built from `transcribe.language`, `code_switch`, `orthography`, `context` and `example`;
  set `transcribe.prompt` to replace it entirely.
- **CER threshold.** Measured on letters only (lowercased, no spaces or punctuation), so casing,
  punctuation and CTC's merged words don't count as disagreement. On Twi/English podcasts most segments
  agree within 15–30%; `max_cer = 50` keeps natural variation in spelling and drops the hallucinations.
- **Digits** are dropped by default: "2019" could have been said in English or the other language, and
  the TTS model must hear what the text says.

## License

MIT. You are responsible for having the right to use the audio you download.
