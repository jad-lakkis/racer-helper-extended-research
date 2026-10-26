#!/usr/bin/env python3
"""
Plot the style vectors written by extract_style_vectors.py (no clustering; see cluster_driver_styles.py).

For every track folder under <style_vectors>/ this writes to <style_vectors>/figures/:
  <track>_corners_map.png            centerline with the detected corners numbered
  <track>_features_vs_time.png       lap-level style features against lap time
  <track>_band_profiles.png          speed / line / brake along the track for each lap-time band
  <track>_time_band_corner_styles.png  mean style vector per corner for each best-lap band, plus the --me driver
  <track>_time_band_vs_<me>.png      band minus the --me driver, per corner
  <track>_style_heatmap.png, <track>_profiles.png   per-lap versions, only for tracks with <= 60 laps
and drivers.csv, driver_style_vectors.csv (one row per driver) next to the data.

Usage (from linesight/):
  python scripts/analyze_style_vectors.py ../style_vectors
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_LAPS = 8  # below this, the across-lap figures say nothing

HEATMAP_PANELS = [
    ("b", "braking fraction b", "viridis", (0, 1)),
    ("f", "full-throttle fraction f", "viridis", (0, 1)),
    ("l_apex", "line at apex (+1 = inside edge)", "coolwarm", (-1, 1)),
    ("s", "steering speed s (full-lock sweeps / s)", "magma", (0, None)),
    ("v_entry_ratio", "entry speed / peak speed", "viridis", (0.3, 1)),
]


def lap_label(row) -> str:
    return f"{row['driver']} {row['lap_time_ms'] / 1000:.2f}s"


def plot_corner_map(track_dir: Path, corners_def: dict, profiles: dict, fig_dir: Path, track: str):
    ref = next(iter(profiles.values()))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(ref["x"], ref["z"], color="0.7", lw=1, label="driven line")
    s = ref["s"]
    for c in corners_def["corners"]:
        m = (s >= c["s_start"]) & (s <= c["s_end"])
        ax.plot(ref["x"][m], ref["z"][m], color="tab:red", lw=3)
        mid = np.argmin(np.abs(s - c["s_apex"]))
        ax.annotate(str(c["idx"]), (ref["x"][mid], ref["z"][mid]), fontsize=10, fontweight="bold", xytext=(6, 6), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_title(f"{track}: {len(corners_def['corners'])} corners on {corners_def['track_length_m']:.0f} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fig.savefig(fig_dir / f"{track}_corners_map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(laps: pd.DataFrame, corners: pd.DataFrame, fig_dir: Path, track: str, order_col: str):
    laps = laps.sort_values(order_col)
    if len(laps) > 40:  # keep the figure readable: fastest 20 + slowest 20
        laps = pd.concat([laps.head(20), laps.tail(20)])
    n_corners = int(corners["corner"].max())
    fig, axes = plt.subplots(len(HEATMAP_PANELS), 1, figsize=(max(8, 0.55 * n_corners + 4), 2.0 * len(HEATMAP_PANELS) + 1), sharex=True)
    for ax, (col, title, cmap, (lo, hi)) in zip(axes, HEATMAP_PANELS):
        mat = corners.pivot(index="lap_id", columns="corner", values=col).reindex(laps["lap_id"]).reindex(columns=range(1, n_corners + 1))
        vmax = hi if hi is not None else max(float(np.nanpercentile(mat.values, 95)), 1.0)
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=lo, vmax=vmax)
        ax.set_yticks(range(len(laps)))
        ax.set_yticklabels([lap_label(r) for _, r in laps.iterrows()], fontsize=7)
        ax.set_title(title, fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    axes[-1].set_xticks(range(n_corners))
    axes[-1].set_xticklabels(range(1, n_corners + 1))
    axes[-1].set_xlabel("corner")
    fig.suptitle(f"{track}: per-corner style vectors", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_style_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(laps: pd.DataFrame, profiles: dict, corners_def: dict, fig_dir: Path, track: str, color_col: str):
    fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 1, 1]))
    cmap = plt.get_cmap("tab10")
    keys = sorted(laps[color_col].unique()) if color_col in laps else [None]
    for _, lap in laps.iterrows():
        p = profiles.get(lap["lap_id"])
        if p is None:
            continue
        col = cmap(keys.index(lap[color_col]) % 10) if color_col in laps else "tab:blue"
        label = f"{color_col} {lap[color_col]}" if color_col in laps else lap_label(lap)
        axes[0].plot(p["s"], p["speed"], color=col, lw=0.9, alpha=0.7, label=label)
        axes[1].plot(p["s"], p["offset"], color=col, lw=0.9, alpha=0.7)
    for lap_i, (_, lap) in enumerate(laps.iterrows()):
        p = profiles.get(lap["lap_id"])
        if p is None:
            continue
        axes[2].fill_between(p["s"], lap_i, lap_i + p["brake"] * 0.9, step="mid", color="tab:red", lw=0)
        axes[3].fill_between(p["s"], lap_i, lap_i + p["accel"] * 0.9, step="mid", color="tab:green", lw=0)
    for ax in axes:
        for c in corners_def["corners"]:
            ax.axvspan(c["s_start"], c["s_end"], color="0.85", alpha=0.5, lw=0)
    for c in corners_def["corners"]:
        axes[0].text(c["s_apex"], axes[0].get_ylim()[1], str(c["idx"]), ha="center", va="top", fontsize=8)
    axes[0].set_ylabel("speed (km/h)")
    axes[1].set_ylabel("offset from centerline (m)")
    axes[2].set_ylabel("brake (1 row/lap)")
    axes[3].set_ylabel("throttle")
    axes[3].set_xlabel("track distance (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0].legend(uniq.values(), uniq.keys(), fontsize=7, ncol=4, loc="lower right")
    fig.suptitle(f"{track}: speed, line, brake and throttle by lap")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_profiles.png", dpi=120, bbox_inches="tight")
    plt.close(fig)








# ─────────────────────────────────────────────────────────────────────────────
# Driver level: one style vector per driver, clustered into groups of drivers
# ─────────────────────────────────────────────────────────────────────────────

DRIVER_FEATURES = ["b", "f", "l_apex", "s", "v_entry_ratio", "brake_onset_m"]
DRIVER_PANELS = [
    ("b", "braking fraction b", "viridis", (0, 0.5)),
    ("f", "full-throttle fraction f", "viridis", (0.5, 1)),
    ("l_apex", "line at apex (+1 = inside edge)", "coolwarm", (-0.6, 0.6)),
    ("s", "steering speed s (full-lock sweeps / s)", "magma", (0, 8)),
    ("v_entry_ratio", "entry speed / peak speed", "viridis", (0.4, 1)),
    ("brake_onset_m", "brake onset before corner (m)", "viridis", (0, 120)),
]


def driver_matrix(laps: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    """One row per driver, columns (feature, corner): mean over all of the driver's valid laps."""
    n_corners = int(corners["corner"].max())
    g = corners[corners["lap_id"].isin(laps["lap_id"])].groupby(["driver", "corner"])[DRIVER_FEATURES].mean().unstack("corner")
    cols = pd.MultiIndex.from_product([DRIVER_FEATURES, range(1, n_corners + 1)])
    return g.reindex(columns=cols)






def plot_group_corner_profiles(groups: dict, wide: pd.DataFrame, fig_dir: Path, track: str, name: str, me: list):
    """Rows = groups of drivers (+ the 'me' rows), columns = corners, one panel per z feature."""
    n_corners = wide.columns.get_level_values(1).max()
    rows = {g: wide.loc[members] for g, members in groups.items() if len(members)}
    mine = [m for m in me if m in wide.index]
    fig, axes = plt.subplots(len(DRIVER_PANELS), 1, figsize=(max(8, 0.55 * n_corners + 4), 1.0 * (len(rows) + len(mine)) * len(DRIVER_PANELS) / 2 + 3), sharex=True)
    for ax, (col, title, cmap, (lo, hi)) in zip(axes, DRIVER_PANELS):
        mat = [rows[g][col].mean().values for g in rows] + [wide.loc[m, col].values for m in mine]
        labels = [f"{g} (n={len(rows[g])})" for g in rows] + [f"{m} (me)" for m in mine]
        im = ax.imshow(np.array(mat, dtype=float), aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    axes[-1].set_xticks(range(n_corners))
    axes[-1].set_xticklabels(range(1, n_corners + 1))
    axes[-1].set_xlabel("corner")
    fig.suptitle(f"{track}: mean style vector per {name}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_{name}_corner_styles.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_me_vs_groups(groups: dict, wide: pd.DataFrame, fig_dir: Path, track: str, name: str, me: str):
    """Group mean minus the 'me' driver, per corner. Red = the group has more of it than me."""
    rows = {g: wide.loc[members] for g, members in groups.items() if len(members)}
    n_corners = wide.columns.get_level_values(1).max()
    fig, axes = plt.subplots(len(DRIVER_PANELS), 1, figsize=(max(8, 0.55 * n_corners + 4), 0.6 * len(rows) * len(DRIVER_PANELS) + 3), sharex=True)
    for ax, (col, title, _, (lo, hi)) in zip(axes, DRIVER_PANELS):
        diff = np.array([rows[g][col].mean().values - wide.loc[me, col].values for g in rows], dtype=float)
        lim = max(float(np.nanpercentile(np.abs(diff), 95)), 1e-6)
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{g} (n={len(rows[g])})" for g in rows], fontsize=7)
        ax.set_title(f"{title}: group minus {me}", fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    axes[-1].set_xticks(range(n_corners))
    axes[-1].set_xticklabels(range(1, n_corners + 1))
    axes[-1].set_xlabel("corner")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_{name}_vs_{me}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_feature_vs_time(laps: pd.DataFrame, fig_dir: Path, track: str, me: list):
    """Lap-level mean of each style feature against lap time. Every dot is a lap."""
    feats = [
        ("mean_b", "braking fraction b"),
        ("mean_f", "full-throttle fraction f"),
        ("mean_l_apex", "line at apex (+1 = inside edge)"),
        ("mean_s", "steering speed s (full-lock sweeps / s)"),
        ("mean_v_entry_ratio", "entry speed / peak speed"),
        ("mean_brake_onset_m", "brake onset before corner (m)"),
    ]
    t = laps["lap_time_ms"] / 1000
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (col, title) in zip(axes.ravel(), feats):
        for kind, color in (("digital", "tab:blue"), ("analog", "tab:orange")):
            m = laps["steer_input"] == kind
            ax.scatter(t[m], laps.loc[m, col], s=10, alpha=0.45, color=color, label=f"{kind} steering ({m.sum()})")
        bins = np.arange(np.floor(t.min()), np.ceil(t.max()) + 1, 1.0)
        mid, med = [], []
        for lo, hi in zip(bins, bins[1:]):
            v = laps.loc[(t >= lo) & (t < hi), col]
            if len(v) >= 5:
                mid.append((lo + hi) / 2)
                med.append(v.median())
        ax.plot(mid, med, color="k", lw=2, label="median per second")
        for name in me:
            m = laps["driver"] == name
            if m.any():
                ax.scatter(t[m], laps.loc[m, col], marker="*", s=260, color="gold", edgecolor="k", zorder=5, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("lap time (s)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{track}: lap-level style features against lap time ({len(laps)} laps, {laps['driver'].nunique()} drivers)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_features_vs_time.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_band_profiles(laps: pd.DataFrame, profiles: dict, corners_def: dict, bands: list, fig_dir: Path, track: str, me: list):
    """Median speed, line and brake probability along the track for each lap-time band, with the 'me' laps on top."""
    t = laps["lap_time_ms"] / 1000
    edges = [0] + list(bands) + [1e9]
    names = [f"{lo:g}-{hi:g}s" if hi < 1e9 else f">{lo:g}s" for lo, hi in zip(edges, edges[1:])]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 2]))
    for c in corners_def["corners"]:
        for ax in axes:
            ax.axvspan(c["s_start"], c["s_end"], color="0.9", lw=0)
        axes[0].text(c["s_apex"], 1.0, str(c["idx"]), transform=axes[0].get_xaxis_transform(), ha="center", va="top", fontsize=8)
    n_groups = len(names)
    for gi, (lo, hi, name) in enumerate(zip(edges, edges[1:], names)):
        ids = laps.loc[(t >= lo) & (t < hi), "lap_id"]
        ps = [profiles[i] for i in ids if i in profiles]
        if not ps:
            continue
        color = cmap(gi / max(n_groups - 1, 1))
        s = ps[0]["s"]
        axes[0].plot(s, np.median([p["speed"] for p in ps], axis=0), color=color, lw=1.4, label=f"{name} (n={len(ps)})")
        axes[1].plot(s, np.median([p["offset"] for p in ps], axis=0), color=color, lw=1.4)
        axes[2].plot(s, np.convolve(np.mean([p["brake"] for p in ps], axis=0), np.ones(5) / 5, mode="same"), color=color, lw=1.2)
    for name in me:
        for i in laps.loc[laps["driver"] == name, "lap_id"]:
            if i in profiles:
                p = profiles[i]
                axes[0].plot(p["s"], p["speed"], color="k", lw=1.2, alpha=0.9, label=name if i == laps.loc[laps["driver"] == name, "lap_id"].iloc[0] else None)
                axes[1].plot(p["s"], p["offset"], color="k", lw=1.2, alpha=0.9)
                axes[2].plot(p["s"], np.convolve(p["brake"].astype(float), np.ones(5) / 5, mode="same"), color="k", lw=1.0, alpha=0.7)
    axes[0].set_ylabel("speed (km/h), median")
    axes[1].set_ylabel("offset from centerline (m), median")
    axes[2].set_ylabel("share of laps braking")
    axes[2].set_xlabel("track distance (m)")
    axes[0].legend(fontsize=8, ncol=3, loc="lower right")
    fig.suptitle(f"{track}: how each lap-time band drives the track (black = {', '.join(me)})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_band_profiles.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def analyse_drivers(track_dir: Path, fig_dir: Path, track: str, laps: pd.DataFrame, corners: pd.DataFrame, me: list, bands: list):
    wide = driver_matrix(laps, corners)
    info = laps.groupby("driver").agg(n_laps=("lap_id", "count"), best_time_s=("lap_time_ms", lambda x: x.min() / 1000))
    info = info.reindex(wide.index)
    edges = [0] + list(bands) + [1e9]
    time_bands = {}
    for lo, hi in zip(edges, edges[1:]):
        members = list(info.index[(info["best_time_s"] >= lo) & (info["best_time_s"] < hi)])
        if members:
            time_bands[f"{lo:g}-{hi:g}s" if hi < 1e9 else f">{lo:g}s"] = members
    print("  drivers per best-lap band:", {k: len(v) for k, v in time_bands.items()})
    info.to_csv(track_dir / "drivers.csv")
    wide.to_csv(track_dir / "driver_style_vectors.csv")
    if len(wide) < 5:
        return
    plot_group_corner_profiles(time_bands, wide, fig_dir, track, "time_band", me)
    for m in [m for m in me if m in wide.index]:
        plot_me_vs_groups(time_bands, wide, fig_dir, track, "time_band", m)


def analyse_track(track_dir: Path, fig_dir: Path, me: list, bands: list):
    track = track_dir.name
    laps = pd.read_csv(track_dir / "laps.csv")
    corners = pd.read_csv(track_dir / "corners.csv")
    corners_def = json.load(open(track_dir / "corners_def.json"))
    profiles = {f.stem: dict(np.load(f)) for f in (track_dir / "profiles").glob("*.npz")}
    print(f"\n== {track}: {len(laps)} laps, {laps['driver'].nunique()} drivers, {len(corners_def['corners'])} corners")

    plot_corner_map(track_dir, corners_def, profiles, fig_dir, track)
    analyse_drivers(track_dir, fig_dir, track, laps, corners, me, bands)
    if len(laps) >= MIN_LAPS and "steer_input" in laps:
        plot_feature_vs_time(laps, fig_dir, track, me)
        plot_band_profiles(laps, profiles, corners_def, bands, fig_dir, track, me)
    if len(laps) <= 60:  # per-lap figures stay readable only for a modest number of laps
        plot_heatmap(laps, corners, fig_dir, track, "lap_time_ms")
        plot_profiles(laps, profiles, corners_def, fig_dir, track, "lap_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("style_dir", type=Path)
    ap.add_argument("--me", default="jadlakkisjad", help="comma-separated driver login(s) to compare against the time bands")
    ap.add_argument("--bands", default="54,56,58,62", help="best-lap band edges in seconds for the time-band comparison")
    args = ap.parse_args()
    fig_dir = args.style_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    tracks = [d for d in sorted(args.style_dir.iterdir()) if (d / "laps.csv").exists()]
    if not tracks:
        sys.exit(f"No <track>/laps.csv under {args.style_dir}; run extract_style_vectors.py first")
    for d in tracks:
        analyse_track(d, fig_dir, [m for m in args.me.split(",") if m], [float(b) for b in args.bands.split(",")])
    print(f"\nfigures in {fig_dir}")


if __name__ == "__main__":
    main()
