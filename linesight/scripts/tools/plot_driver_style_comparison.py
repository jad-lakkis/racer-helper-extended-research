#!/usr/bin/env python3
"""
Compare the style vectors of a few drivers on one track: per-corner profiles, lap-level summary bars, and a table.

Reads <style_dir>/<track>/{laps.csv,corners.csv,corners_def.json,profiles/} written by extract_style_vectors.py and writes
  figures/<track>_driver_styles.png      per-corner b, f, line, steering speed, brake point, entry speed for every driver
  figures/<track>_driver_summary.png     the same measures averaged over the corners
  <track>/driver_style_table.csv         the lap-level numbers behind the bars
  figures/<track>_corners_map.png        the track with its corners numbered

  python scripts/tools/plot_driver_style_comparison.py ../style_vectors/Interlagos_GP
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_style_vectors import plot_corner_map  # noqa: E402

PANELS = [
    ("b", "braking fraction b (share of the corner spent braking)", (0, 1)),
    ("f", "full-throttle fraction f", (0, 1)),
    ("l_apex", "line at the apex (+1 = inside edge, relative to the fastest driver's line)", None),
    ("s", "steering speed s (full-lock sweeps per second)", None),
    ("bz_onset_m", "brake point: metres before the corner entry (blank = not braked)", None),
    ("v_entry_ratio", "entry speed / the lap's peak speed", (0.1, 1.05)),
]
MIN_BRAKED = 3
SUMMARY = [("b", "braking fraction b"), ("f", "full-throttle fraction f"), ("l_apex", "line at apex"), ("s", "steering speed s"), ("bz_onset_m", "brake point (m before entry)"), ("v_entry_ratio", "entry speed / peak")]


def display_name(file: str, track: str) -> str:
    m = re.match(rf"^{re.escape(track)}_(.+)_(\d+)_(\d+)\.Replay\.Gbx$", file)
    return m.group(1) if m else file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track_dir", type=Path)
    args = ap.parse_args()
    track = args.track_dir.name
    fig_dir = args.track_dir.parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    laps = pd.read_csv(args.track_dir / "laps.csv").sort_values("lap_time_ms")
    corners = pd.read_csv(args.track_dir / "corners.csv")
    corners_def = json.load(open(args.track_dir / "corners_def.json"))
    profiles = {f.stem: dict(np.load(f)) for f in (args.track_dir / "profiles").glob("*.npz")}
    laps["name"] = [display_name(f, track) for f in laps["file"]]
    laps["label"] = [f"{r.name}  {r.lap_time_ms / 1000:.1f} s, {r.steer_input} steering" for r in laps.itertuples()]
    colors = dict(zip(laps["lap_id"], plt.get_cmap("tab10").colors))
    n_corners = int(corners["corner"].max())

    plot_corner_map(args.track_dir, corners_def, profiles, fig_dir, track)

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(max(9, 0.8 * n_corners + 5), 2.5 * len(PANELS)), sharex=True)
    for ax, (col, title, ylim) in zip(axes, PANELS):
        for r in laps.itertuples():
            g = corners[corners["lap_id"] == r.lap_id].sort_values("corner")
            ax.plot(g["corner"], g[col], marker="o", ms=5, lw=1.4, color=colors[r.lap_id], label=r.label)
        ax.set_title(title, fontsize=9, loc="left")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    axes[-1].set_xticks(range(1, n_corners + 1))
    axes[-1].set_xlabel("corner (numbered along the lap, see the map)")
    fig.suptitle(f"{track}: style vector of each driver, corner by corner", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_driver_styles.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    table = []
    for r in laps.itertuples():
        g = corners[corners["lap_id"] == r.lap_id]
        row = dict(driver=r.name, lap_time_s=round(r.lap_time_ms / 1000, 2), steering=r.steer_input, braked_corners=int(g["braked"].sum()))
        for col, _ in SUMMARY:
            row[col] = round(float(g[col].mean()), 3) if not (col.startswith("bz_") and g["braked"].sum() < MIN_BRAKED) else np.nan
        row["brake_len_m"] = round(float(g["bz_len_m"].mean()), 1) if g["braked"].sum() >= MIN_BRAKED else np.nan
        table.append(row)
    table = pd.DataFrame(table)
    table.to_csv(args.track_dir / "driver_style_table.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (col, title) in zip(axes.ravel(), SUMMARY):
        vals = []
        for r in laps.itertuples():
            g = corners[corners["lap_id"] == r.lap_id]
            # a mean brake point over one or two braked corners says little: show it only for drivers who braked in at least MIN_BRAKED corners
            vals.append(float(g[col].mean()) if not (col.startswith("bz_") and g["braked"].sum() < MIN_BRAKED) else np.nan)
        ax.bar(range(len(laps)), np.nan_to_num(vals), color=[colors[i] for i in laps["lap_id"]])
        ax.set_xticks(range(len(laps)))
        ax.set_xticklabels([f"{n}\n{t / 1000:.1f}s" for n, t in zip(laps["name"], laps["lap_time_ms"])], fontsize=8)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, 0 if np.isnan(v) else v, f"braked in {int(corners[corners['lap_id'] == laps['lap_id'].iloc[i]]['braked'].sum())} corners" if np.isnan(v) else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{track}: style of each driver, averaged over the corners", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_driver_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
