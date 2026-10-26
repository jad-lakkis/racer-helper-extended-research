#!/usr/bin/env python3
"""
Download replays for one track from a Trackmania Exchange site (default: TMNF-X).

Replays are saved as <TrackName>_<user>_<time>_<replayid>.Replay.Gbx so that
extract_style_vectors.py recognises the track from the filename, plus a manifest.csv with the
site's metadata (replay id, uploader, time, date, validated flag, respawns).

Usage:
  python scripts/tools/fetch_tmx_replays.py 414041 --out ../replays/tmnf_exchange/Hockolicious
  python scripts/tools/fetch_tmx_replays.py 414041 --max-time 65 --per-band 15   # sample of each time band

Requirements: pip install requests
"""

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HOST = "https://tmnf.exchange"
BANDS_S = [0, 50, 52, 54, 56, 58, 60, 65, 70, 80]


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track_id", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--max-time", type=float, default=80.0, help="skip replays slower than this many seconds")
    ap.add_argument("--allow-respawns", action="store_true", help="also keep replays with respawns")
    ap.add_argument("--per-band", type=int, help="keep at most this many replays per time band (spread over distinct drivers)")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between downloads")
    args = ap.parse_args()

    s = requests.Session()
    track = s.get(f"{args.host}/api/tracks", params=dict(id=args.track_id, fields="TrackId,TrackName"), timeout=30).json()["Results"]
    if not track:
        sys.exit(f"track {args.track_id} not found on {args.host}")
    track_name = track[0]["TrackName"]

    rows, after = [], None
    while True:
        params = dict(
            trackId=args.track_id,
            count=1000,
            fields="ReplayId,User.Name,User.UserId,ReplayTime,ReplayAt,ReplayScore,Position,Validated,ReplayRespawns",
        )
        if after:
            params["after"] = after
        page = s.get(f"{args.host}/api/replays", params=params, timeout=60).json()
        rows += page["Results"]
        if not page["More"] or not page["Results"]:
            break
        after = page["Results"][-1]["ReplayId"]

    df = pd.DataFrame(
        dict(
            replay_id=[r["ReplayId"] for r in rows],
            user=[r["User"]["Name"] for r in rows],
            user_id=[r["User"]["UserId"] for r in rows],
            time_ms=[r["ReplayTime"] for r in rows],
            replay_at=[r["ReplayAt"] for r in rows],
            validated=[r["Validated"] for r in rows],
            respawns=[r["ReplayRespawns"] for r in rows],
        )
    ).drop_duplicates("replay_id")
    print(f"{track_name}: {len(df)} replays listed")

    df = df[df["time_ms"] <= args.max_time * 1000]
    if not args.allow_respawns:
        df = df[df["respawns"] == 0]
    if args.per_band:
        picks = []
        for lo, hi in zip(BANDS_S, BANDS_S[1:]):
            band = df[(df["time_ms"] >= lo * 1000) & (df["time_ms"] < hi * 1000)].sort_values("time_ms")
            band = band.drop_duplicates("user_id")  # best replay of each driver first
            if len(band) > args.per_band:
                band = band.iloc[[int(i) for i in pd.Series(range(args.per_band)) * len(band) / args.per_band]]
            picks.append(band)
        df = pd.concat(picks)
    df = df.sort_values("time_ms")
    print(f"downloading {len(df)} replays to {args.out}")

    args.out.mkdir(parents=True, exist_ok=True)
    files = []
    for r in df.itertuples():
        name = f"{safe(track_name)}_{safe(r.user)}_{r.time_ms}_{r.replay_id}.Replay.Gbx"
        path = args.out / name
        files.append(name)
        if path.exists() and path.stat().st_size > 0:
            continue
        resp = s.get(f"{args.host}/recordgbx/{r.replay_id}", timeout=60)
        if resp.status_code != 200 or not resp.content:
            print(f"  failed {r.replay_id}: HTTP {resp.status_code}")
            files[-1] = ""
            continue
        path.write_bytes(resp.content)
        time.sleep(args.delay)
    df = df.assign(file=files)
    df[df["file"] != ""].to_csv(args.out / "manifest.csv", index=False)
    print(f"done: {(df['file'] != '').sum()} files, manifest at {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
