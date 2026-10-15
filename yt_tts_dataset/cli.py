"""yt-tts-dataset: YouTube playlist / video URLs -> single-speaker, transcribed TTS dataset."""
import argparse
import subprocess
import sys
import time

from . import config

STAGES = ["diarize", "segment", "screen", "demucs", "transcribe", "crosscheck"]


def enabled(cfg, stage):
    return {"screen": cfg["screen"]["enabled"], "demucs": cfg["demucs"]["enabled"], "crosscheck": cfg["crosscheck"]["enabled"]}.get(stage, True)


def run_stage(cfg, stage, follow=False, **kw):
    if stage == "diarize":
        from .diarize import run
    elif stage == "segment":
        from .segment import run
    elif stage == "screen":
        from .screen import run
    elif stage == "demucs":
        from .demucs_vocals import run
    elif stage == "transcribe":
        from .transcribe import run
    elif stage == "crosscheck":
        from .crosscheck import run
    run(cfg, follow=follow, **kw)


def run_all(cfg, args):
    if not args.skip_download:
        from .download import run as download
        download(cfg)
    stages = [s for s in STAGES if enabled(cfg, s)]
    if args.sequential:
        for s in stages:
            print(f"== {s}", flush=True)
            run_stage(cfg, s)
    else:  # every stage in its own process, each picking up episodes as the previous stage finishes them
        logs = cfg.dir("logs")
        procs = []
        for s in stages:
            log = open(logs / f"{s}.log", "a")
            procs.append((s, subprocess.Popen([sys.executable, "-m", "yt_tts_dataset.cli", "-c", str(cfg.path), s,
                                               "--follow"], stdout=log, stderr=subprocess.STDOUT)))
            print(f"{s}: started (log {logs / f'{s}.log'})", flush=True)
        while any(p.poll() is None for _, p in procs):
            for s, p in procs:
                if p.poll():
                    for _, q in procs:
                        q.terminate()
                    sys.exit(f"{s} failed, see {logs / f'{s}.log'}")
            time.sleep(10)
    from .build import report
    report(cfg)
    print("\ncheck the table above, set build.max_cer, then run `yt-tts-dataset build`")


def main():
    ap = argparse.ArgumentParser(prog="yt-tts-dataset", description=__doc__)
    ap.add_argument("-c", "--config", default="config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="every stage, then the filter report")
    p.add_argument("--sequential", action="store_true", help="one stage after another instead of all at once")
    p.add_argument("--skip-download", action="store_true")
    p = sub.add_parser("download", help="expand playlists into videos.tsv and download the audio")
    p.add_argument("--skip-expand", action="store_true", help="reuse videos.tsv")
    for s in STAGES:
        p = sub.add_parser(s)
        p.add_argument("--follow", action="store_true", help="keep waiting for the previous stage until all done")
        if s == "diarize":
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--nshards", type=int, default=1)
        if s == "transcribe":
            p.add_argument("--retry-failed", action="store_true", help="re-request segments that came back null")
    sub.add_parser("report", help="hours kept at each filter setting")
    sub.add_parser("build", help="apply [build] filters and write dataset/")
    args = ap.parse_args()
    cfg = config.load(args.config)

    if args.cmd == "run":
        run_all(cfg, args)
    elif args.cmd == "download":
        from .download import run
        run(cfg, skip_expand=args.skip_expand)
    elif args.cmd == "report":
        from .build import report
        report(cfg)
    elif args.cmd == "build":
        from .build import run
        run(cfg)
    elif args.cmd == "transcribe" and args.retry_failed:
        from .transcribe import retry_failed
        retry_failed(cfg)
    elif args.cmd == "diarize":
        run_stage(cfg, "diarize", args.follow, shard=args.shard, nshards=args.nshards)
    else:
        run_stage(cfg, args.cmd, args.follow)


if __name__ == "__main__":
    main()
