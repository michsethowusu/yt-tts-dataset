"""Stage 1: expand playlists/URLs into data/videos.tsv (id, seconds, title), filter by length, and
download best audio with yt-split-downloader (github.com/michsethowusu/yt-split-downloader).

yt-split-downloader resolves the media URLs on this machine (a trusted IP YouTube serves) and downloads
the bytes where the backend says:
  local   on this machine                         -> audio in <data_dir>/yt/videos
  ssh     on a server (e.g. a GPU box whose datacenter IP YouTube blocks), pulled back here too unless
          download.pull = false;
          run the other stages on the server with download.audio_dir = <remote_dir>/videos
  modal   on Modal, fetched back here
This stage keeps resolving until every video has arrived, or nothing new has arrived for stall_minutes.
"""
import subprocess
import sys
import time
from pathlib import Path


def expand(cfg):
    src = cfg["source"]
    urls = list(src["urls"])
    if src["urls_file"]:
        urls += [l.strip() for l in cfg.resolve(src["urls_file"]).read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
    if not urls:
        sys.exit("no source.urls / source.urls_file in the config")
    seen, rows = set(), []
    for url in urls:
        out = subprocess.run(["yt-dlp", "--flat-playlist", "--ignore-errors", "--print",
                              "%(id)s\t%(duration)s\t%(title)s", url], capture_output=True, text=True).stdout
        for line in out.splitlines():
            vid, dur, title = (line.split("\t", 2) + ["", ""])[:3]
            if vid and vid not in seen:
                seen.add(vid)
                rows.append((vid, float(dur) if dur not in ("", "NA", "None") else 0.0, title.replace("\t", " ")))
    lo, hi = src["min_minutes"] * 60, src["max_minutes"] * 60
    keep = [r for r in rows if r[1] >= lo and (not hi or r[1] <= hi)]
    with open(cfg.data / "videos.tsv", "w") as f:
        f.writelines(f"{v}\t{d:.0f}\t{t}\n" for v, d, t in keep)
    print(f"{len(rows)} videos found, {len(keep)} kept ({sum(r[1] for r in keep) / 3600:.1f} h) "
          f"-> {cfg.data / 'videos.tsv'}")
    return [r[0] for r in keep]


def ytsd_config(cfg, ids):
    """Build a yt-split-downloader Config from [download]."""
    from yt_split_downloader.config import Config

    d = cfg["download"]
    yc = Config()
    project = cfg.dir("yt")
    (project / "urls.txt").write_text("".join(f"https://www.youtube.com/watch?v={v}\n" for v in ids))
    yc.general.project_dir = str(project)
    yc.general.urls_file = str(project / "urls.txt")
    yc.general.format = d["format"]
    yc.general.cookies = d["cookies"] or None
    yc.download.workers = d["workers"]
    yc.download.connections = d["connections"]
    yc.backend.type = d["backend"]
    yc.backend.host = d["host"] or None
    yc.backend.remote_dir = d["remote_dir"] or None
    yc.backend.remote_python = d["remote_python"]
    return yc


def run(cfg, skip_expand=False):
    from yt_split_downloader.backends import get_backend
    from yt_split_downloader.layout import Layout
    from yt_split_downloader.resolver import run_batch, run_stream

    cfg.data.mkdir(parents=True, exist_ok=True)
    tsv, miss = cfg.data / "videos.tsv", cfg.data / "videos_missing.tsv"
    if skip_expand and tsv.exists():
        if miss.exists():  # give earlier failures another try
            with open(tsv, "a") as f:
                f.write(miss.read_text())
            miss.unlink()
        # yt-split-downloader waits 6 h before re-resolving an id; retry missing ones now
        (cfg.dir("yt") / "attempts.json").unlink(missing_ok=True)
        ids = cfg.ids()
    else:
        ids = expand(cfg)
    yc = ytsd_config(cfg, ids)
    backend = get_backend(yc)
    backend.prepare()
    layout = Layout(yc.project_dir)
    if yc.backend.type == "modal":
        run_batch(yc, backend, once=True, fetch=True)
    else:
        want, last, last_t = set(ids), -1, time.time()
        while True:
            run_stream(yc, backend, once=True)  # one resolve pass; the download worker keeps running
            done = want & (layout.read_archive() | backend.remote_done())
            if len(done) != last:
                print(f"downloaded {len(done)}/{len(want)}", flush=True)
                last, last_t = len(done), time.time()
            if done == want or time.time() - last_t > 60 * cfg["download"]["stall_minutes"]:
                break
            time.sleep(yc.general.poll_interval)
        if yc.backend.type == "ssh" and cfg["download"]["pull"]:
            backend.pull_results(layout)
    if yc.backend.type == "ssh" and not cfg["download"]["pull"]:  # files stay on the server
        missing = [v for v in ids if v not in done]
    else:
        missing = [v for v in ids if not audio_path(cfg, v)]
    if missing:  # set aside so later stages don't wait for them
        print(f"{len(missing)} not downloaded (rerun `download --skip-expand` to retry): {' '.join(missing)}")
        lines = tsv.read_text().splitlines(keepends=True)
        miss.write_text("".join(l for l in lines if l.split("\t")[0] in missing))
        tsv.write_text("".join(l for l in lines if l.split("\t")[0] not in missing))


def audio_path(cfg, vid):
    hits = [p for p in Path(cfg.audio_dir).glob(f"{vid}.*") if not p.name.endswith((".part", ".tmp", ".aria2"))]
    return hits[0] if hits else None
