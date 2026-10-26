#!/usr/bin/env python3
"""
Cluster and plot the style vectors written by extract_style_vectors.py.

For every track folder under <style_vectors>/ this writes (to <style_vectors>/figures/):
  <track>_corners_map.png    centerline with the detected corners numbered
  <track>_style_heatmap.png  one row per lap, one column per corner, one panel per z component
  <track>_profiles.png       speed / lateral line / brake / throttle against track distance
  <track>_style_space.png    PCA of the lap-level style vectors     (needs >= MIN_LAPS laps)
  <track>_cluster_cards.png  what each style cluster looks like     (needs >= MIN_LAPS laps)
and lap_clusters.csv + clusters.json (cluster -> RL conditioning values) next to the data.

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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_LAPS = 8  # below this, clustering says nothing, so only per-lap style cards are drawn
MAX_K = 6

# lap-level features that define "style" (lap time itself is deliberately not one of them)
STYLE_FEATURES = [
    "mean_b",
    "mean_f",
    "mean_l_entry",
    "mean_l_apex",
    "mean_l_exit",
    "mean_s",
    "mean_b_approach",
    "mean_brake_onset_m",
    "mean_v_entry_ratio",
    "oversteer_understeer_score",
]
CONDITIONING = ["braking_aggression", "risk_tolerance_proxy", "oversteer_understeer_score", "corner_entry_speed_ratio"]
HEATMAP_PANELS = [
    ("b", "braking fraction b", "viridis", (0, 1)),
    ("f", "full-throttle fraction f", "viridis", (0, 1)),
    ("l_apex", "line at apex (+1 = inside edge)", "coolwarm", (-1, 1)),
    ("s", "steering edges / s", "magma", (0, None)),
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


def cluster_laps(laps: pd.DataFrame):
    feats = laps[STYLE_FEATURES].copy()
    feats = feats.fillna(feats.median())
    z = StandardScaler().fit_transform(feats)
    best = None
    for k in range(2, min(MAX_K, len(laps) // 3) + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(z)
        score = silhouette_score(z, km.labels_)
        if best is None or score > best[0]:
            best = (score, k, km.labels_)
    return z, best


def plot_style_space(laps: pd.DataFrame, z: np.ndarray, fig_dir: Path, track: str, silhouette: float):
    pca = PCA(n_components=2).fit(z)
    xy = pca.transform(z)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    drivers = sorted(laps["driver"].unique())
    markers = "osD^vP*Xh<>"
    cmap = plt.get_cmap("tab10")
    for i, (d, g) in enumerate(laps.groupby("driver")):
        idx = [laps.index.get_loc(j) for j in g.index]
        ax.scatter(xy[idx, 0], xy[idx, 1], c=[cmap(int(c) % 10) for c in g["style_cluster"]], marker=markers[drivers.index(d) % len(markers)], s=45, edgecolor="k", lw=0.4, label=d)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title(f"{track}: laps in style space (colour = cluster, marker = driver), silhouette {silhouette:.2f}")
    if len(drivers) <= 12:
        ax.legend(fontsize=7, title="driver")
    fig.savefig(fig_dir / f"{track}_style_space.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_cards(laps: pd.DataFrame, fig_dir: Path, track: str):
    ks = sorted(laps["style_cluster"].unique())
    z = (laps[STYLE_FEATURES] - laps[STYLE_FEATURES].mean()) / laps[STYLE_FEATURES].std(ddof=0).replace(0, 1)
    cent = z.groupby(laps["style_cluster"]).mean()
    fig, axes = plt.subplots(1, len(ks), figsize=(3.2 * len(ks) + 2, 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, ks):
        v = cent.loc[k].values
        ax.barh(range(len(STYLE_FEATURES)), v, color=["tab:red" if x > 0 else "tab:blue" for x in v])
        ax.set_yticks(range(len(STYLE_FEATURES)))
        ax.set_yticklabels([f.replace("mean_", "") for f in STYLE_FEATURES], fontsize=8)
        ax.axvline(0, color="k", lw=0.6)
        sub = laps[laps["style_cluster"] == k]
        ax.set_title(f"cluster {k}: {len(sub)} laps, {sub['driver'].nunique()} drivers\nmedian {sub['lap_time_ms'].median() / 1000:.2f}s", fontsize=9)
        ax.set_xlim(-2.5, 2.5)
    axes[0].invert_yaxis()
    fig.supxlabel("z-score vs all laps on this track")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_cluster_cards.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def analyse_track(track_dir: Path, fig_dir: Path):
    track = track_dir.name
    laps = pd.read_csv(track_dir / "laps.csv")
    corners = pd.read_csv(track_dir / "corners.csv")
    corners_def = json.load(open(track_dir / "corners_def.json"))
    profiles = {f.stem: dict(np.load(f)) for f in (track_dir / "profiles").glob("*.npz")}
    print(f"\n== {track}: {len(laps)} laps, {laps['driver'].nunique()} drivers, {len(corners_def['corners'])} corners")

    plot_corner_map(track_dir, corners_def, profiles, fig_dir, track)
    order = "lap_time_ms"
    summary = {}

    if len(laps) >= MIN_LAPS:
        z, best = cluster_laps(laps)
        silhouette, k, labels = best
        laps["style_cluster"] = labels
        order = "style_cluster"
        plot_style_space(laps, z, fig_dir, track, silhouette)
        plot_cluster_cards(laps, fig_dir, track)
        print(f"  {k} style clusters (silhouette {silhouette:.2f})")
        summary["clusters"] = {}
        for c, g in laps.groupby("style_cluster"):
            summary["clusters"][int(c)] = dict(
                n_laps=len(g),
                drivers=sorted(g["driver"].unique()),
                median_lap_time_ms=float(g["lap_time_ms"].median()),
                style=g[STYLE_FEATURES].mean().round(3).to_dict(),
                rl_conditioning=g[CONDITIONING].mean().round(3).to_dict(),
            )
        # how much of each style feature is just "being fast"
        corr = laps[STYLE_FEATURES].corrwith(laps["lap_time_ms"], method="spearman").round(2)
        summary["spearman_with_lap_time"] = corr.to_dict()
        print("  Spearman vs lap time:", corr.to_dict())
        laps[["lap_id", "driver", "lap_time_ms", "style_cluster"]].to_csv(track_dir / "lap_clusters.csv", index=False)
        json.dump(summary, open(track_dir / "clusters.json", "w"), indent=2)
        profile_color = "style_cluster"
    else:
        print(f"  only {len(laps)} laps (< {MIN_LAPS}): no clustering, per-lap style cards only")
        profile_color = "lap_id"

    plot_heatmap(laps, corners, fig_dir, track, order)
    plot_profiles(laps, profiles, corners_def, fig_dir, track, profile_color)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("style_dir", type=Path)
    args = ap.parse_args()
    fig_dir = args.style_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    tracks = [d for d in sorted(args.style_dir.iterdir()) if (d / "laps.csv").exists()]
    if not tracks:
        sys.exit(f"No <track>/laps.csv under {args.style_dir}; run extract_style_vectors.py first")
    for d in tracks:
        analyse_track(d, fig_dir)
    print(f"\nfigures in {fig_dir}")


if __name__ == "__main__":
    main()
