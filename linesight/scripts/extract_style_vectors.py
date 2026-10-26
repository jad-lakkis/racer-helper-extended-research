#!/usr/bin/env python3
"""
Per-corner driver style vectors from Trackmania .Replay.Gbx files.

For every valid lap and every corner of the track this builds the style vector from the
problem formulation (z = (b, f, l, s)):

  b  fraction of corner time spent braking
  f  fraction of corner time at full throttle (accelerate pressed, brake released)
  l  driven line: lateral offset from the track centerline at corner entry / apex / exit,
     divided by half the track width and signed so that +1 is the inside edge of the corner
  s  steering speed: total variation of the steering command (full-lock sweeps per second) inside
     the corner. Digital presses count as full-lock jumps, analog pad / wheel input is continuous

plus extra per-corner features (entry / apex / min speed, brake onset before the corner,
coasting, steering reversals) used by analyze_style_vectors.py.

Corners are defined once per track from the centerline curvature, so every lap and every driver
is measured on the same segments and corner k is comparable across laps.

Usage (from linesight/):
  python scripts/extract_style_vectors.py ../replays --out ../style_vectors
  python scripts/extract_style_vectors.py ../replays/some_dir --track-name MyTrack \
      --centerline maps/MyTrack_0.5m_cl.npy --out ../style_vectors

Requirements: pip install pygbx scipy pandas numpy
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_driver_profile import (  # noqa: E402
    _load_ghost,
    decode_analog_steer,
    _pos_curvature,
    compute_braking_aggression,
    compute_corner_entry_speed,
    compute_oversteer_understeer,
)

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from trackmania_rl.style_features import (  # noqa: E402,F401  (re-exported: other scripts import these names from here)
    PERIOD_MS,
    Centerline,
    Corner,
    LapSignals,
    _brake_zone,
    _reversals_u,
    build_centerline,
    corner_rows,
    detect_corners,
    project_on_centerline,
)

# filename keyword -> (track name, centerline file under linesight/maps)
KNOWN_TRACKS = {
    "bahrain": ("Bahrain", "Bahrain_Circuit_0.5m_jadlakkisjad_012629.npy"),
    "hockolicious": ("Hockolicious", "ESL-Hockolicious_0.5m_cl2.npy"),
}

DEFAULT_TRACK_WIDTH_M = 32.0  # TMNF road block width; assumption, override with --track-width
PROFILE_STEP_M = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Replay -> track coordinates and input timelines
# ─────────────────────────────────────────────────────────────────────────────


def _timeline(control_entries, name: str, race_start_ms: int, n: int) -> np.ndarray:
    evts = sorted([(e.time - race_start_ms, bool(e.enabled)) for e in control_entries if e.event_name == name])
    state = np.zeros(n, dtype=bool)
    cur, ptr = False, 0
    for i in range(n):
        while ptr < len(evts) and evts[ptr][0] <= i * PERIOD_MS:
            cur = evts[ptr][1]
            ptr += 1
        state[i] = cur
    return state


def _steer_signal(control_entries, race_start_ms: int, n: int):
    """
    Steering command u in [-1, 1] on the 10 ms grid, held between events, plus the input type.
    Replays store steering either as digital SteerLeft/SteerRight presses or as 'Steer' events
    (analog value for pad / wheel drivers, -1/0/+1 for keyboard drivers on newer replays).
    + = right in both. (In the replay files the analog value is positive for a left turn, checked on 40 keyboard laps against the
    path they drove; the features in corners.csv only use |u| and its changes, so they do not depend on the sign.)
    """
    steer = sorted([(e.time - race_start_ms, decode_analog_steer(e)) for e in control_entries if e.event_name == "Steer"])
    if steer:
        u, cur, ptr = np.zeros(n), 0.0, 0
        for i in range(n):
            while ptr < len(steer) and steer[ptr][0] <= i * PERIOD_MS:
                cur = steer[ptr][1]
                ptr += 1
            u[i] = -cur  # replay 'Steer' values have the opposite sign to SteerLeft/SteerRight: this makes + = right, as for keyboard presses
        analog = any(min(abs(v), abs(abs(v) - 1.0)) > 0.02 for _, v in steer)
        return np.clip(u, -1.0, 1.0), "analog" if analog else "digital"
    left = _timeline(control_entries, "SteerLeft", race_start_ms, n)
    right = _timeline(control_entries, "SteerRight", race_start_ms, n)
    return right.astype(float) - left.astype(float), "digital"


# ─────────────────────────────────────────────────────────────────────────────
# One lap
# ─────────────────────────────────────────────────────────────────────────────


def analyse_lap(path: Path, track: str, cl: Centerline, corners: list, half_width: float, approach_m: float):
    ghost, race_start_ms, race_time_ms, n_samples = _load_ghost(str(path))
    if ghost.num_respawns:
        return None, None, f"{ghost.num_respawns} respawn(s)"

    recs = ghost.records[: n_samples + 1]
    if len(recs) < 20:
        return None, None, f"only {len(recs)} position records"
    positions = np.array([[r.position.x, r.position.y, r.position.z] for r in recs])
    disp = np.array([r.display_speed for r in recs], dtype=np.float64)  # km/h
    t_s = np.arange(len(recs)) * 0.1
    t_s = np.minimum(t_s, race_time_ms / 1000.0)

    p, e = project_on_centerline(positions, disp / 3.6, cl)
    p = np.maximum.accumulate(p)
    if p[-1] < 0.9 * cl.s[-1]:
        return None, None, f"lap covers only {p[-1] / cl.s[-1]:.0%} of the centerline"
    if np.median(np.abs(e)) > 0.6 * half_width:
        return None, None, "driven line far from the centerline (different version of the track?)"

    ce = ghost.control_entries
    n10 = race_time_ms // PERIOD_MS + 1
    brake = _timeline(ce, "Brake", race_start_ms, n10)
    accel = _timeline(ce, "Accelerate", race_start_ms, n10)
    u, steer_input = _steer_signal(ce, race_start_ms, n10)
    sig = LapSignals(t_s=t_s, p=p, e=e, speed_kmh=disp, brake=brake, accel=accel, u=u, race_time_ms=race_time_ms)
    p_strict, e10, v10, peak_v = sig.p_strict, sig.e10, sig.v10, sig.peak_v

    lap_id = f"{track}_{re.sub(r'[^A-Za-z0-9._-]+', '-', ghost.login)}_{race_time_ms}"  # also a filename
    meta = dict(lap_id=lap_id, track=track, driver=ghost.login, lap_time_ms=race_time_ms)
    rows = corner_rows(sig, corners, half_width, approach_m, meta)

    # lap level: z averaged over corners plus the global scalars used as RL conditioning
    pos_curv = _pos_curvature(positions[:n_samples], disp[:n_samples], n_samples)
    braking = compute_braking_aggression(ce, race_start_ms, race_time_ms, n_samples, disp[:n_samples], pos_curv)
    ous = compute_oversteer_understeer(positions[:n_samples], disp[:n_samples], ce, race_start_ms, n_samples)
    _, corner_ratio = compute_corner_entry_speed(disp[:n_samples], n_samples, pos_curv)
    df = pd.DataFrame(rows)
    lap = dict(
        lap_id=lap_id,
        track=track,
        driver=ghost.login,
        lap_time_ms=race_time_ms,
        file=path.name,
        steer_input=steer_input,
        n_corners=int(df["b"].notna().sum()) if "b" in df else 0,
        peak_speed_kmh=peak_v,
        mean_speed_kmh=float(v10.mean()),
        braking_aggression=braking,
        oversteer_understeer_score=ous,
        corner_entry_speed_ratio=corner_ratio,
        mean_abs_offset_m=float(np.abs(e10).mean()),
        max_abs_offset_m=float(np.abs(e10).max()),
        risk_tolerance_proxy=float(np.clip(np.abs(e10).mean() / 15.0, 0, 1)),
    )
    for k in ("b", "f", "l_entry", "l_apex", "l_exit", "s", "b_approach", "brake_onset_m", "v_entry_ratio", "coast"):
        lap[f"mean_{k}"] = float(df[k].mean()) if k in df else np.nan

    # profile on a fixed distance grid for plotting
    grid = np.arange(0.0, cl.s[-1], PROFILE_STEP_M)
    gi = np.clip(np.round(np.interp(grid, p_strict, t_s) * 1000 / PERIOD_MS).astype(int), 0, n10 - 1)
    profile = dict(
        s=grid,
        speed=v10[gi],
        offset=e10[gi],
        brake=brake[gi],
        accel=accel[gi],
        steer=u[gi].astype(np.float32),
        x=np.interp(grid, p_strict, positions[: len(p_strict), 0]),
        z=np.interp(grid, p_strict, positions[: len(p_strict), 2]),
    )
    return (lap, rows), profile, None


# ─────────────────────────────────────────────────────────────────────────────
# Style ranges (problem formulation: ranges of z from the K fastest valid laps)
# ─────────────────────────────────────────────────────────────────────────────

Z_COLS = ["b", "f", "l_entry", "l_apex", "l_exit", "s"]


def style_ranges(laps: pd.DataFrame, corners: pd.DataFrame, k: int) -> dict:
    out = {}
    for driver, dl in laps.groupby("driver"):
        best = dl.nsmallest(k, "lap_time_ms")
        sub = corners[corners["lap_id"].isin(best["lap_id"])]
        entry = dict(n_laps=len(best), lap_times_ms=best["lap_time_ms"].tolist(), corners={})
        for cidx, g in sub.groupby("corner"):
            entry["corners"][int(cidx)] = {
                col: dict(min=float(g[col].min()), max=float(g[col].max()), mean=float(g[col].mean()), std=float(g[col].std(ddof=0)))
                for col in Z_COLS
                if g[col].notna().any()
            }
        out[driver] = entry
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def guess_track(path: Path):
    low = path.name.lower()
    for key, (name, cl_file) in KNOWN_TRACKS.items():
        if key in low:
            return name, BASE_DIR / "maps" / cl_file
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replays", type=Path, help="a .Replay.Gbx file or a directory searched recursively")
    ap.add_argument("--out", type=Path, default=BASE_DIR.parent / "style_vectors")
    ap.add_argument("--track-name", help="force the track name (needed with --centerline)")
    ap.add_argument("--centerline", type=Path, help="(n, 3) .npy centerline, 0.5 m spacing")
    ap.add_argument("--track-width", type=float, default=DEFAULT_TRACK_WIDTH_M, help="road width in metres used to normalise the line")
    ap.add_argument("--approach-m", type=float, default=120.0, help="distance before a corner scanned for brake onset")
    ap.add_argument("--top-k", type=int, default=5, help="fastest valid laps per driver used for the style ranges")
    ap.add_argument("--corner-curv", type=float, default=1 / 150, help="centerline curvature threshold (1/m) for corners")
    ap.add_argument("--corner-min-angle", type=float, default=20.0)
    args = ap.parse_args()

    files = [args.replays] if args.replays.is_file() else sorted(args.replays.rglob("*.Replay.Gbx"))
    if not files:
        sys.exit(f"No .Replay.Gbx files under {args.replays}")

    per_track = {}
    for f in files:
        if args.track_name:
            name, cl_path = args.track_name, args.centerline
        else:
            name, cl_path = guess_track(f)
        if name is None or cl_path is None:
            print(f"skip {f.name}: unknown track, pass --track-name and --centerline")
            continue
        per_track.setdefault(name, (cl_path, []))[1].append(f)

    for name, (cl_path, track_files) in per_track.items():
        cl = build_centerline(np.load(cl_path))
        corners = detect_corners(cl, k_thresh=args.corner_curv, min_angle_deg=args.corner_min_angle)
        out = args.out / name
        (out / "profiles").mkdir(parents=True, exist_ok=True)
        print(f"\n== {name}: {len(corners)} corners on {cl.s[-1]:.0f} m, {len(track_files)} replay(s)")

        lap_rows, corner_rows, skipped = [], [], []
        for f in track_files:
            try:
                result, profile, why = analyse_lap(f, name, cl, corners, args.track_width / 2, args.approach_m)
            except SystemExit as exc:
                result, profile, why = None, None, str(exc)
            except Exception as exc:  # one malformed replay must not stop a batch of hundreds
                result, profile, why = None, None, f"{type(exc).__name__}: {exc}"
            if result is None:
                print(f"  skip {f.name}: {why}")
                skipped.append(dict(file=f.name, reason=why))
                continue
            lap, rows = result
            lap_rows.append(lap)
            corner_rows.extend(rows)
            np.savez_compressed(out / "profiles" / f"{lap['lap_id']}.npz", **profile)
            print(f"  {f.name}: {lap['driver']} {lap['lap_time_ms'] / 1000:.2f}s")

        pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(out / "skipped.csv", index=False)
        if not lap_rows:
            continue
        laps = pd.DataFrame(lap_rows).drop_duplicates("lap_id").sort_values("lap_time_ms")
        cdf = pd.DataFrame(corner_rows)
        cdf = cdf[cdf["lap_id"].isin(laps["lap_id"])].drop_duplicates(["lap_id", "corner"])
        laps.to_csv(out / "laps.csv", index=False)
        cdf.to_csv(out / "corners.csv", index=False)
        json.dump(
            dict(
                track=name,
                centerline=str(cl_path),
                track_length_m=float(cl.s[-1]),
                track_width_m=args.track_width,
                corners=[c.__dict__ for c in corners],
            ),
            open(out / "corners_def.json", "w"),
            indent=2,
        )
        json.dump(style_ranges(laps, cdf, args.top_k), open(out / "style_ranges.json", "w"), indent=2)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
