#!/usr/bin/env python3
"""
Compare the layout of a TMNF track (from the fastest replay on tmnf.exchange) with the real circuit outline
(bacinger/f1-circuits, OpenStreetMap based), both drawn from above with the corners numbered in lap order.

  python scripts/tools/compare_tmnf_to_real.py --tmnf-track 5169102 --real it-1922 --name Monza --out ../tracks/compare

TMNF coordinates are drawn with z pointing down; the best fit may still use a mirror, which is reported.
Needs network access to tmnf.exchange and raw.githubusercontent.com. The shape comparison tries every start point, both
driving directions and a mirror image, and fits a rotation and a single scale factor (orthogonal Procrustes).
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_driver_profile import _load_ghost  # noqa: E402
from extract_style_vectors import build_centerline, detect_corners  # noqa: E402

TMX = "https://tmnf.exchange"
REAL = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/circuits/{}.geojson"
SMOOTH_M = 8.0


def resample_closed(p: np.ndarray, step: float = None, n: int = None) -> np.ndarray:
    q = np.vstack([p, p[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    t = np.arange(0, s[-1], step) if step else np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)


def length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum())


def smooth_loop(p: np.ndarray, sigma_m: float, step: float = 2.0) -> np.ndarray:
    p = resample_closed(p, step)
    return np.stack([gaussian_filter1d(p[:, i], sigma_m / step, mode="wrap") for i in range(2)], 1)


def first_lap(P: np.ndarray, speeds_ms: np.ndarray):
    """Positions of the first lap: up to the first return to the start after leaving it; the whole path if it never returns."""
    d = np.linalg.norm(P[:, [0, 2]] - P[0, [0, 2]], axis=1)
    left = np.argmax(d > 150) if (d > 150).any() else None
    if left is not None:
        back = np.where((d[left:] < 40))[0]
        if len(back):
            return P[: left + back[0] + 1], True
    return P, False


def tmnf_loop(track_id: int):
    rep = requests.get(f"{TMX}/api/replays", params=dict(trackId=track_id, count=5, fields="ReplayId,ReplayTime,ReplayRespawns,User.Name"), timeout=30).json()["Results"]
    rep = [r for r in rep if r["ReplayRespawns"] == 0][0]
    blob = requests.get(f"{TMX}/recordgbx/{rep['ReplayId']}", timeout=60).content
    f = tempfile.NamedTemporaryFile(suffix=".Replay.Gbx", delete=False)
    f.write(blob)
    f.close()
    try:
        g, _, rt, n = _load_ghost(f.name)
        P = np.array([[r.position.x, r.position.y, r.position.z] for r in g.records[: n + 1]])
        spd = np.array([r.display_speed for r in g.records[: n + 1]]) / 3.6
    finally:
        os.unlink(f.name)
    lap, closed = first_lap(P, spd)
    return lap, closed, dict(replay_id=rep["ReplayId"], time_s=rep["ReplayTime"] / 1000, driver=rep["User"]["Name"], full_path_km=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / 1000))


def real_loop(real_id: str):
    d = requests.get(REAL.format(real_id), timeout=30).json()
    feat = d["features"][0]
    ll = np.array(feat["geometry"]["coordinates"])[:, :2]
    lat0 = ll[:, 1].mean()
    xy = np.stack([(ll[:, 0] - ll[:, 0].mean()) * 111320 * np.cos(np.radians(lat0)), (ll[:, 1] - ll[:, 1].mean()) * 110574], 1)
    return xy, feat["properties"]


def align(A: np.ndarray, B: np.ndarray, n: int = 200):
    """Best similarity transform of loop A onto loop B over start point, direction and mirror. Returns (A aligned, scale, rms_m, mirrored, reversed)."""
    a, b = resample_closed(A, n=n), resample_closed(B, n=n)
    bc = b - b.mean(0)
    rb = np.sqrt((bc**2).sum(1).mean())
    best = None
    for mirror in (False, True):
        for rev in (False, True):
            a2 = a * np.array([-1.0 if mirror else 1.0, 1.0])
            a2 = a2[::-1] if rev else a2
            ac = a2 - a2.mean(0)
            ra = np.sqrt((ac**2).sum(1).mean())
            for shift in range(n):
                x = np.roll(ac, shift, axis=0)
                U, S, Vt = np.linalg.svd(x.T @ bc)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    continue
                s = S.sum() / (x**2).sum()
                err = np.sqrt((((s * x @ R) - bc) ** 2).sum(1).mean())
                if best is None or err < best[0]:
                    best = (err, shift, mirror, rev, R, s)
    err, shift, mirror, rev, R, s = best
    full = resample_closed(A, n=n) * np.array([-1.0 if mirror else 1.0, 1.0])
    full = full[::-1] if rev else full
    full = np.roll(full - full.mean(0), shift, axis=0)
    return s * full @ R + b.mean(0), float(s), float(err), mirror, rev


def corners_of(loop_m: np.ndarray):
    p = smooth_loop(loop_m, SMOOTH_M)
    cl = build_centerline(np.column_stack([p[:, 0], np.zeros(len(p)), p[:, 1]]), smooth_m=SMOOTH_M)
    return p, cl, detect_corners(cl, k_thresh=1 / 250, min_angle_deg=15, min_len_m=15, merge_gap_m=15)


def draw(ax, p, cl, corners, title, color):
    ax.plot(np.append(p[:, 0], p[0, 0]), np.append(p[:, 1], p[0, 1]), color=color, lw=2)
    ax.plot(*p[0], "k*", ms=14)
    ax.annotate("", xy=p[10], xytext=p[0], arrowprops=dict(arrowstyle="->", lw=2))
    for c in corners:
        i = int(np.argmin(np.abs(cl.s - c.s_apex)))
        ax.annotate(str(c.idx), p[i], xytext=(6, 6), textcoords="offset points", fontsize=10, fontweight="bold", color="tab:red")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tmnf-track", type=int, required=True)
    ap.add_argument("--real", required=True, help="circuit id in bacinger/f1-circuits, e.g. it-1922")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lap, closed, info = tmnf_loop(args.tmnf_track)
    t_xy = lap[:, [0, 2]] * np.array([1.0, -1.0])  # TM's z axis points down the page in a map view, so flip it to see the layout the right way up
    r_xy, props = real_loop(args.real)
    aligned, scale, rms, mirrored, rev = align(t_xy, r_xy)
    # TMNF loop at real-world scale (before any rotation), so corner thresholds mean the same on both
    t_scaled = t_xy * scale
    tp, tcl, tcorners = corners_of(t_scaled)
    rp, rcl, rcorners = corners_of(r_xy)
    tmnf_len_m = length(smooth_loop(t_scaled, SMOOTH_M))
    real_len_m = float(props.get("length", length(r_xy)))

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    draw(axes[0], tp, tcl, tcorners, f"TMNF track {args.tmnf_track}: {len(tcorners)} corners, path {length(lap[:, [0, 2]]) / 1000:.2f} km (scale x{scale:.2f})", "tab:blue")
    draw(axes[1], rp, rcl, rcorners, f"real {args.real}: {len(rcorners)} corners, {real_len_m / 1000:.3f} km", "tab:green")
    ax = axes[2]
    ax.plot(np.append(r_xy[:, 0], r_xy[0, 0]), np.append(r_xy[:, 1], r_xy[0, 1]), color="tab:green", lw=2.5, label="real")
    ax.plot(np.append(aligned[:, 0], aligned[0, 0]), np.append(aligned[:, 1], aligned[0, 1]), color="tab:blue", lw=1.8, label="TMNF, best fit")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    ax.set_title(f"overlay: RMS gap {rms:.0f} m ({100 * rms / real_len_m:.1f} % of the lap), {'mirrored, ' if mirrored else ''}{'reversed' if rev else 'same direction'}", fontsize=10)
    fig.suptitle(f"{args.name}: TMNF replay of {info['driver']} ({info['time_s']:.1f} s) against the real outline", y=1.0)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / f"{args.name}_{args.tmnf_track}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(args.out / f"{args.name}_{args.tmnf_track}.npz", real=r_xy, tmnf_aligned=aligned, tmnf_corners=np.array([[c.idx, *tp[int(np.argmin(np.abs(tcl.s - c.s_apex)))]] for c in tcorners]), real_corners=np.array([[c.idx, *rp[int(np.argmin(np.abs(rcl.s - c.s_apex)))]] for c in rcorners]), tmnf_xy=tp, real_xy=rp)
    row = dict(name=args.name, tmnf_track=args.tmnf_track, real=args.real, first_lap_found=bool(closed), scale=round(scale, 3), tmnf_len_at_real_scale_km=round(tmnf_len_m / 1000, 3),
               real_len_km=round(real_len_m / 1000, 3), rms_gap_m=round(rms, 1), rms_pct_of_lap=round(100 * rms / real_len_m, 2), mirrored=mirrored,
               corners_tmnf=len(tcorners), corners_real=len(rcorners), **info)
    print(json.dumps(row))
    return row


if __name__ == "__main__":
    main()
