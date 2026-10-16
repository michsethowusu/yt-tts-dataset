# yt-tts-dataset

Turn a YouTube playlist (or any list of video URLs) into a text-to-speech dataset: full-band,
single-speaker segments with verbatim transcripts, built for conversational and code-switched speech.

```
playlist / URLs ──► download ──► diarize ──► segment ──► screen ──► transcribe ──► crosscheck ──► build
                    yt-split-    pyannote    single-     PANNs      Gemini or      omniASR CTC     filter +
                    downloader   3.1         speaker     non-speech Google ASR     second opinion  train/val
                                             3–15 s      score      (optional:     (optional)
                                                                    demucs first)
```

Built at [Ghana NLP](https://ghananlp.org) for Twi/English podcasts, but the language, prompt and
thresholds are all in the config.

## What each stage does

| Stage | Output (`data/…`) | Why |
|---|---|---|
| **download** | `videos.tsv`, `yt/videos/<id>.mp4` | Expands playlists, drops videos outside `min_minutes`/`max_minutes` (e.g. short excerpts of full episodes), downloads best audio only (opus, 48 kHz) with [yt-split-downloader](https://github.com/michsethowusu/yt-split-downloader). |
| **diarize** | `flac/<id>.flac`, `diar/<id>.rttm` | 48 kHz mono FLAC master; speaker turns from pyannote 3.1 (default), NVIDIA Sortformer (GPU, much faster) or sherpa-onnx (CPU). Used **only** to keep segments single-speaker — no speaker IDs end up in the dataset. |
| **segment** | `segs/<id>/*.flac`, `segs/<id>.jsonl` | Keeps each speaker's exclusive regions (other speakers' turns subtracted, so no overlap or crosstalk), cuts at Silero VAD pauses into 3–15 s segments. Pieces of one speaker are joined across short pauses only if nobody else speaks in the pause. |
| **screen** | `screen/<id>.jsonl` | [PANNs](https://github.com/qiuqiangkong/panns_inference) sound event detection (AudioSet's 527 classes, 10 ms frames) scores how much of anything other than speech each segment holds: music, laughter, applause, singing, crowd, noise. `build` drops segments above `max_nonspeech`. Fast, and it drops the segment instead of trying to clean it. On by default. |
| **demucs** | `vocals/<id>/*.flac` | Optional (`demucs.enabled`). htdemucs vocals stem: removes music beds, jingles and background noise. Back to the master sample rate. Slow; for sources with music under the speech. |
| **transcribe** | `text/<id>.jsonl` | `engine = "gemini"`: many requests in flight; verbatim, keeps casing and punctuation (leave normalisation to your TTS framework), numbers written as words. `engine = "google"`: Google's free speech endpoint in one language (`google_language`, e.g. `en-GH`), no key needed, no punctuation. |
| **crosscheck** | `xcheck/<id>.jsonl` | omniASR CTC-1B transcribes the same audio. An LLM can hallucinate fluent text on music or unclear audio; a CTC model can't invent words that aren't there. CER between the two (letters only) flags bad transcripts. |
| **build** | `dataset/{train,val}.jsonl`, `metadata_{train,val}.csv` | Drops empty transcripts and segments above `max_nonspeech` or `max_cer`, with digits, with implausible characters/second, or whose speaking rate is an outlier; holds out whole episodes for validation. |

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
yt-tts-dataset report                  # hours kept at each filter setting — look before you drop
yt-tts-dataset build                   # apply [build] filters, write data/dataset/
```

`run` starts every stage as its own process (logs in `data/logs/`); each picks up episodes as the
stage before it finishes them, so the GPU stages and the transcription requests overlap. `run --sequential`
does one stage at a time. Stages can also be run alone, e.g. `yt-tts-dataset transcribe --follow`,
`yt-tts-dataset segment --shard 0 --nshards 2` (any stage can run as several processes side by side,
each taking every n-th episode), or
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
  The downloads are copied back to your PC as well, unless you set `download.pull = false`: do that
  when the PC has little disk space, since the audio is only needed on the server.
- `modal` — download on [Modal](https://modal.com), fetched back here.

If some videos fail, `yt-tts-dataset download --skip-expand` retries just those.

## Options

All in `config.toml`; `config.example.toml` lists every key with its default.

### Diarization: pyannote, Sortformer, sherpa-onnx or none

| | `engine = "pyannote"` (default) | `engine = "sortformer"` | `engine = "sherpa"` |
|---|---|---|---|
| Model | speaker-diarization-3.1 | NVIDIA Streaming Sortformer 4spk v2 | pyannote segmentation-3.0 + WeSpeaker, ONNX |
| Runs on | GPU, clustering on CPU | GPU only, end to end | CPU (or CUDA build) |
| Speed (H200, busy CPU) | 10–17× realtime per process | ~170× realtime | ~11× |
| Limits | HF token, gated model | ≤ 4 speakers; NeMo install | tune `cluster_threshold` |

Measured on a 101-minute two-person interview: Sortformer found the same host and guest as pyannote
(69 vs 76 min and 19 vs 19 min) at 17% DER against it, mostly a little less speech marked. Since
diarization only decides which regions are single-speaker, that is enough; use pyannote where you
want its accuracy and the CPU is free.

NeMo pins many packages, so install it in its own venv and point `diarize.python` at it:

```bash
python -m venv .venv-nemo && .venv-nemo/bin/pip install "nemo_toolkit[asr]" soundfile
```

```toml
[diarize]
engine = "sortformer"
python = ".venv-nemo/bin/python"
```

**One speaker throughout** (a solo channel, a lecture series): skip diarization with `engine = "none"`.
Each episode becomes one speaker turn covering the whole file, so `segment` cuts at VAD pauses only;
it takes seconds and needs no GPU or token. Other voices (an intro, a quoted clip) are not removed,
and episodes shorter than `segment.min_speaker_minutes` are skipped.

```toml
[diarize]
engine = "none"
```

### Transcription: Gemini or Google

| | `engine = "gemini"` (default) | `engine = "google"` |
|---|---|---|
| Text | Verbatim, casing and punctuation, numbers as words, code-switching in each language's spelling | Lowercase-ish, no punctuation, one language per run |
| Setup | `GEMINI_API_KEY`; `model`, `thinking`, `concurrency`, prompt fields | `google_language` (BCP-47, e.g. `"en-GH"`, `"en-US"`, `"ak"`), `workers` threads; no key |
| Cost | Paid (see Notes) | Free (unofficial endpoint, can rate-limit) |
| Best for | Code-switched speech, when exact text matters | Single-language speech, large volumes |

```toml
[transcribe]
engine = "google"
google_language = "en-GH"
workers = 16
audio = "segments"    # or "vocals" to transcribe the Demucs output
```

With Google, a segment it can't make out comes back empty (`""`) and `build` drops it. That makes the
engine a quality filter too: on interviews, the empty ones were mostly unclear speech, crosstalk or
another language. A request that failed (network, rate limit) is `null` instead; `transcribe
--retry-failed` re-requests those.

### Cleaning the audio: screen or Demucs

| | `[screen]` (on by default) | `[demucs]` (off by default) |
|---|---|---|
| Does | Scores each segment for any sound that isn't speech; `build` **drops** the bad ones | Separates the vocals and **keeps** the segment, background removed |
| Speed | Seconds per episode | Minutes per episode |
| Use when | The source is mostly clean speech (interviews, podcasts) and you'd rather lose a segment than keep a laugh or a jingle | Music plays under the speech for long stretches and dropping those parts would lose too much |

```toml
[screen]
enabled = true
batch = 64          # halved automatically on CUDA out-of-memory
smooth = 0.3        # seconds a sound must last to count
allow = ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
         "Conversation", "Narration, monologue", "Silence", "Inside, small room"]

[demucs]
enabled = false
```

`allow` holds the [AudioSet class names](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/metadata/class_labels_indices.csv)
that don't count as non-speech; everything else does (laughter, music, applause, singing, crowd,
animals, noise…). Both can be on: the screen runs on the original segments, Demucs output is what
goes into the dataset.

### Filters in `build`

| Key | Default | Drops |
|---|---|---|
| `max_nonspeech` | `0.3` | segments whose non-speech score (from `screen`) is above it |
| `rate_outlier_z` | `"auto"` | segments whose speaking rate is an outlier (below); `"auto"` = 3 for google, off for gemini; `0` = off |
| `min_cps`, `max_cps` | `4`, `30` | characters per second outside this fixed range |
| `max_cer` | `50` | disagreement with the crosscheck ASR above this % (only with `crosscheck.enabled`) |
| `drop_digits` | `true` | transcripts containing digits (their spoken form is ambiguous) |
| `spell_digits` | `""` | (not a filter) a [num2words](https://github.com/savoirfairelinux/num2words) language code, e.g. `"en"`: writes numbers out as words first (years, ordinals, %, $/£/€, decimals), so only leftovers are dropped |
| `val_episodes` | `2` | (not a filter) whole episodes held out for validation |

Empty transcripts are always dropped. Run `yt-tts-dataset report` before `build`: it prints the
hours each setting keeps, which sounds the screen would drop, and how much the rate filter removes as
too slow or too fast, so you can pick thresholds from your own data.

## Output

`data/dataset/train.jsonl` (and `val.jsonl`), one segment per line:

```json
{"audio": "data/segs/<id>/<id>_0042.flac", "text": "Yeah, nti wohwɛ a, na ɛnyɛ production no a yɛreyi wɔ Kumase no",
 "dur": 3.9, "episode": "<id>", "start": 812.31, "end": 816.21, "rate_z": -0.4, "nonspeech": 0.02,
 "nonspeech_label": "Music", "source_audio": "data/segs/<id>/<id>_0042.flac"}
```

`audio` is the Demucs vocals (`data/vocals/…`) when Demucs is on. `cer` and `omni` are added with the
crosscheck, `removed_db` with Demucs.

`metadata_train.csv` has `audio|text` lines for frameworks that take LJSpeech-style lists.

## Notes

- **Cost.** Gemini 3.8 Flash with low thinking used ≈ 200k input + 30k output tokens per hour of audio.
- **Google engine.** Uses the unofficial free endpoint behind SpeechRecognition's `recognize_google`:
  no key or cost, but one language per run, no punctuation, and it can rate-limit you (failed requests
  are kept as `null`; `transcribe --retry-failed` re-requests them). Lower `workers` if many fail.
- **Prompt.** Built from `transcribe.language`, `code_switch`, `orthography`, `context` and `example`;
  set `transcribe.prompt` to replace it entirely.
- **Non-speech threshold.** `nonspeech` is the highest score (0–1, averaged over `screen.smooth`
  seconds) of any AudioSet class not in `screen.allow`; `nonspeech_label` names it. `report` shows
  the hours kept at each threshold and which sounds the dropped segments had.
- **Speaking-rate outliers.** Google's free endpoint sometimes returns only part of what was said, so
  the text is far too short for its audio. `build` measures each segment's rate in letters per second
  (no spaces or punctuation, so Google's and Gemini's text compare alike), takes the robust z-score of
  its log against the whole dataset (median and 1.4826 × MAD) and drops segments more than
  `rate_outlier_z` deviations away, too slow or too fast. Unlike `min_cps`/`max_cps` it adapts to how
  fast the people in your source actually talk.
- **CER threshold.** Measured on letters only (lowercased, no spaces or punctuation), so casing,
  punctuation and CTC's merged words don't count as disagreement. On Twi/English podcasts most segments
  agree within 15–30%; `max_cer = 50` keeps natural variation in spelling and drops the hallucinations.
- **Digits** are dropped by default: "2019" could have been said in English or the other language, and
  the TTS model must hear what the text says.

## License

MIT. You are responsible for having the right to use the audio you download.
