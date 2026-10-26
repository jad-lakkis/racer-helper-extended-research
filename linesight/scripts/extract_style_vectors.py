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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

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

# filename keyword -> (track name, centerline file under linesight/maps)
KNOWN_TRACKS = {
    "bahrain": ("Bahrain", "Bahrain_Circuit_0.5m_jadlakkisjad_012629.npy"),
    "hockolicious": ("Hockolicious", "ESL-Hockolicious_0.5m_cl2.npy"),
}

DEFAULT_TRACK_WIDTH_M = 32.0  # TMNF road block width; assumption, override with --track-width
PERIOD_MS = 10  # resolution of the input timelines (ghost positions are 100 ms apart)
PROFILE_STEP_M = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Corners from the centerline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Corner:
    idx: int
    s_start: float
    s_end: float
    s_apex: float
    direction: int  # +1: centre of curvature on the +normal side, -1: other side
    angle_deg: float
    min_radius_m: float


@dataclass
class Centerline:
    pts: np.ndarray  # (n, 3)
    s: np.ndarray  # arc length at every point
    tangent: np.ndarray  # (n, 2) unit tangent in the x-z plane
    normal: np.ndarray  # (n, 2) tangent rotated +90 degrees
    heading: np.ndarray  # smoothed unwrapped heading
    curvature: np.ndarray  # signed 1/m, > 0 turns towards +normal


def build_centerline(pts: np.ndarray, smooth_m: float = 6.0) -> Centerline:
    xz = pts[:, [0, 2]]
    seg = np.diff(xz, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    ds = float(seg_len.mean())
    heading = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    heading = np.concatenate([heading, heading[-1:]])
    heading_s = gaussian_filter1d(heading, smooth_m / ds, mode="nearest")
    curvature = np.gradient(heading_s, s)
    tangent = np.stack([np.cos(heading_s), np.sin(heading_s)], axis=1)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return Centerline(pts, s, tangent, normal, heading_s, curvature)


def detect_corners(
    cl: Centerline,
    k_thresh: float = 1 / 150,
    merge_gap_m: float = 25.0,
    min_len_m: float = 20.0,
    min_angle_deg: float = 20.0,
) -> list:
    """Contiguous same-sign runs of |curvature| > k_thresh, merged across short gaps."""
    kappa = cl.curvature
    sign = np.where(kappa > k_thresh, 1, np.where(kappa < -k_thresh, -1, 0))
    runs = []  # [start_idx, end_idx, sign]
    i, n = 0, len(sign)
    while i < n:
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sign[j + 1] == sign[i]:
            j += 1
        runs.append([i, j, int(sign[i])])
        i = j + 1

    merged = []
    for r in runs:
        if merged and merged[-1][2] == r[2] and cl.s[r[0]] - cl.s[merged[-1][1]] < merge_gap_m:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    corners = []
    for a, b, sg in merged:
        length = cl.s[b] - cl.s[a]
        angle = abs(np.degrees(cl.heading[b] - cl.heading[a]))
        if length < min_len_m or angle < min_angle_deg:
            continue
        apex = a + int(np.argmax(np.abs(kappa[a : b + 1])))
        corners.append(
            Corner(
                idx=len(corners) + 1,
                s_start=float(cl.s[a]),
                s_end=float(cl.s[b]),
                s_apex=float(cl.s[apex]),
                direction=sg,
                angle_deg=float(angle),
                min_radius_m=float(1.0 / max(abs(kappa[apex]), 1e-6)),
            )
        )
    return corners


# ─────────────────────────────────────────────────────────────────────────────
# Replay -> track coordinates and input timelines
# ─────────────────────────────────────────────────────────────────────────────


def project_on_centerline(positions: np.ndarray, speeds_ms: np.ndarray, cl: Centerline, dt: float = 0.1):
    """
    Sequential projection: each sample may only advance a bounded distance past the previous
    one, so sections of the track that pass close to each other cannot be confused.
    Returns arc length p and signed lateral offset e (+ = along cl.normal).
    """
    n = len(positions)
    ds = float(cl.s[-1] / (len(cl.s) - 1))
    xz = positions[:, [0, 2]]
    cxz = cl.pts[:, [0, 2]]
    p = np.zeros(n)
    e = np.zeros(n)
    prev = 0
    for i in range(n):
        reach = int((speeds_ms[i] * dt * 3.0 + 25.0) / ds)
        lo, hi = max(0, prev - int(10 / ds)), min(len(cxz), prev + reach + 1)
        d = np.linalg.norm(cxz[lo:hi] - xz[i], axis=1)
        k = lo + int(np.argmin(d))
        rel = xz[i] - cxz[k]
        p[i] = cl.s[k] + float(rel @ cl.tangent[k])
        e[i] = float(rel @ cl.normal[k])
        prev = k
    return p, e


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


def _reversals_u(u: np.ndarray, deadband: float = 0.1) -> int:
    d = np.sign(u[np.abs(u) > deadband])
    return int(np.count_nonzero(np.diff(d) != 0)) if len(d) > 1 else 0



BRAKE_WINDOW_M = 200.0  # how far before a corner entry a brake press still belongs to that corner
BRAKE_CHAIN_GAP = 40  # samples (0.4 s): presses closer than this are one braking zone
BRAKE_MIN_SCRUB_KMH = 8.0  # a zone must take off at least this much speed and be at least BRAKE_MIN_LEN_M long
BRAKE_MIN_LEN_M = 10.0


def _brake_zone(brake, p10, v10, c, prev_end, idx_at, peak_v) -> dict:
    """
    The braking zone of one corner: among the chains of brake presses between max(entry - BRAKE_WINDOW_M,
    end of the previous corner) and the apex, the one that takes off the most speed.
      bz_onset_m    metres before the corner entry where braking starts
      bz_release_m  metres after the corner entry where braking ends (> 0 means braking into the corner, trail braking)
      bz_len_m      length of the zone
      bz_presses    separate presses inside the zone (1 = one clean press, more = pumping)
      bz_scrub_kmh  speed lost between onset and release
      bz_onset_speed speed at onset / peak speed of the lap
    All NaN when the corner is not braked (no chain takes off 8 km/h over 10 m); bz_taps_other counts the other chains.
    """
    out = dict(braked=0, bz_onset_m=np.nan, bz_release_m=np.nan, bz_len_m=np.nan, bz_presses=np.nan, bz_scrub_kmh=np.nan, bz_onset_speed=np.nan, bz_taps_other=0.0)
    lo, hi = idx_at(max(c.s_start - BRAKE_WINDOW_M, prev_end)), idx_at(c.s_apex)
    seg = brake[lo : hi + 1].astype(np.int8)
    if not seg.any():
        return out
    d = np.diff(np.concatenate([[0], seg, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]  # run i covers samples starts[i]..ends[i]-1
    chains, first = [], 0
    for r in range(1, len(starts) + 1):
        if r == len(starts) or starts[r] - ends[r - 1] >= BRAKE_CHAIN_GAP:
            chains.append((first, r - 1))
            first = r
    best = None
    for f, l in chains:
        i_on, i_off = lo + starts[f], lo + ends[l] - 1
        scrub = float(v10[i_on] - v10[i_off])
        if best is None or scrub > best[0]:
            best = (scrub, f, l, i_on, i_off)
    scrub, f, l, i_on, i_off = best
    length = float(p10[i_off] - p10[i_on])
    out["bz_taps_other"] = float(sum(ends[ll] - starts[ff] > 0 for ff, ll in chains) - 1)  # other chains in the window
    if scrub < BRAKE_MIN_SCRUB_KMH or length < BRAKE_MIN_LEN_M:
        return out  # a tap, not a braking zone
    out.update(
        braked=1,
        bz_onset_m=float(c.s_start - p10[i_on]),
        bz_release_m=float(p10[i_off] - c.s_start),
        bz_len_m=length,
        bz_presses=float(l - f + 1),
        bz_scrub_kmh=scrub,
        bz_onset_speed=float(v10[i_on]) / peak_v,
    )
    return out


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

    n10 = race_time_ms // PERIOD_MS + 1
    t10 = np.arange(n10) * PERIOD_MS / 1000.0
    p_strict = p + np.arange(len(p)) * 1e-6  # strictly increasing for the inverse interpolation
    p10 = np.interp(t10, t_s, p_strict)
    e10 = np.interp(t10, t_s, e)
    v10 = np.interp(t10, t_s, disp)

    ce = ghost.control_entries
    brake = _timeline(ce, "Brake", race_start_ms, n10)
    accel = _timeline(ce, "Accelerate", race_start_ms, n10)
    u, steer_input = _steer_signal(ce, race_start_ms, n10)
    peak_v = max(float(v10.max()), 1.0)

    def t_at(s_pos: float) -> float:
        return float(np.interp(s_pos, p_strict, t_s))

    def idx_at(s_pos: float) -> int:
        return int(np.clip(round(t_at(s_pos) * 1000 / PERIOD_MS), 0, n10 - 1))

    def e_at(s_pos: float) -> float:
        return float(np.interp(s_pos, p10, e10))

    lap_id = f"{track}_{re.sub(r'[^A-Za-z0-9._-]+', '-', ghost.login)}_{race_time_ms}"  # also a filename
    rows = []
    for ci, c in enumerate(corners):
        ia, ib, iap = idx_at(c.s_start), idx_at(c.s_end), idx_at(c.s_apex)
        row = dict(lap_id=lap_id, track=track, driver=ghost.login, lap_time_ms=race_time_ms, corner=c.idx)
        if ib - ia < 3:
            rows.append(row)
            continue
        dur = (ib - ia) * PERIOD_MS / 1000.0
        sl = slice(ia, ib)
        sgn = c.direction / half_width
        row.update(
            b=float(brake[sl].mean()),
            f=float((accel[sl] & ~brake[sl]).mean()),
            l_entry=e_at(c.s_start) * sgn,
            l_apex=e_at(c.s_apex) * sgn,
            l_exit=e_at(c.s_end) * sgn,
            l_mean=float(e10[sl].mean()) * sgn,
            s=float(np.abs(np.diff(u[ia : ib + 1])).sum()) / dur,
            steer_frac=float((np.abs(u[sl]) > 0.05).mean()),
            steer_reversals=_reversals_u(u[sl]),
            coast=float((~accel[sl] & ~brake[sl]).mean()),
            v_entry=float(v10[ia]),
            v_apex=float(v10[iap]),
            v_min=float(v10[sl].min()),
            v_exit=float(v10[ib]),
            v_entry_ratio=float(v10[ia]) / peak_v,
            corner_time_s=dur,
        )
        # braking before the corner: how early and how much
        pa = idx_at(max(c.s_start - approach_m, 0.0))
        app = brake[pa:ia]
        row["b_approach"] = float(app.mean()) if len(app) else np.nan
        if app.any():
            first = pa + int(np.argmax(app))
            row["brake_onset_m"] = float(c.s_start - np.interp(first * PERIOD_MS / 1000.0, t_s, p_strict))
        else:
            row["brake_onset_m"] = 0.0
        row.update(_brake_zone(brake, p10, v10, c, corners[ci - 1].s_end if ci else 0.0, idx_at, peak_v))
        rows.append(row)

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
