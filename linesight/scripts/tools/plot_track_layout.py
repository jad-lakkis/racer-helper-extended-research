#!/usr/bin/env python3
"""
Plot a track from centerline.npy (x, y, z, 0.5 m apart): top-down map coloured by elevation with the corners numbered,
elevation profile, and signed curvature along the lap.

Usage (from linesight/):
  python scripts/tools/plot_track_layout.py ../tracks/monza/centerline.npy --title "Monza (Trackmania map 271697)" --out ../tracks/monza/layout.png
  add --labels "1-2:Rettifilo,3-4:Roggia,5:Lesmo 1,6:Lesmo 2,7-9:Ascari,10:Parabolica" to name corner groups
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_style_vectors import build_centerline, detect_corners  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("centerline", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--labels", default="", help='corner group names, e.g. "1-2:Rettifilo,10:Parabolica"')
    ap.add_argument("--flip-z", action="store_true", help="draw z downwards (use when the lap should run clockwise on the page)")
    ap.add_argument("--corner-curv", type=float, default=1 / 250)
    ap.add_argument("--corner-min-angle", type=float, default=15.0)
    args = ap.parse_args()

    pts = np.load(args.centerline)
    cl = build_centerline(pts)
    corners = detect_corners(cl, k_thresh=args.corner_curv, min_angle_deg=args.corner_min_angle, min_len_m=15, merge_gap_m=15)
    names = {}
    for item in filter(None, args.labels.split(",")):
        k, name = item.split(":")
        a, _, b = k.partition("-")
        for i in range(int(a), int(b or a) + 1):
            names[i] = name

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], height_ratios=[1, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    sc = ax.scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], s=4, cmap="terrain")
    fig.colorbar(sc, ax=ax, label="elevation (m)", shrink=0.6)
    ax.plot(pts[0, 0], pts[0, 2], "k*", ms=18, label="start")
    ax.annotate("", xy=pts[200, [0, 2]], xytext=pts[0, [0, 2]], arrowprops=dict(arrowstyle="->", lw=2))
    for c in corners:
        i = int(np.argmin(np.abs(cl.s - c.s_apex)))
        ax.annotate(str(c.idx), pts[i, [0, 2]], xytext=(8, 8), textcoords="offset points", fontsize=11, fontweight="bold", color="tab:red")
    ax.set_aspect("equal")
    if args.flip_z:
        ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"{args.title}: {cl.s[-1]:.0f} m, {len(corners)} corners")
    ax.legend(loc="upper left")

    s = cl.s
    bands = lambda a: [a.axvspan(c.s_start, c.s_end, color="tab:red", alpha=0.12, lw=0) for c in corners]  # noqa: E731
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(s, pts[:, 1], color="k", lw=1)
    a1.set_ylabel("elevation (m)")
    bands(a1)
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(s, cl.curvature * 1000, color="tab:blue", lw=1)
    a2.set_ylabel("signed curvature (1/km)")
    a2.axhline(0, color="k", lw=0.5)
    bands(a2)
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    for c in corners:
        a3.bar(c.s_apex, c.min_radius_m, width=40, color="tab:red" if c.direction > 0 else "tab:green")
        a3.text(c.s_apex, c.min_radius_m + 3, str(c.idx) + (f"\n{names[c.idx]}" if c.idx in names else ""), ha="center", fontsize=8)
    a3.set_ylabel("tightest radius (m)")
    a3.set_xlabel("distance along the lap (m)   red = turn one way, green = the other")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"{len(corners)} corners, {cl.s[-1]:.1f} m")


if __name__ == "__main__":
    main()
