#!/usr/bin/env python3
"""
What changes when the same driver gets faster?

TMX keeps a driver's improving personal bests, so many drivers have several replays of the same track at
different times and dates. For those drivers this script
  1. compares the style features of their slowest and fastest lap (per second gained) with the
     difference between faster and slower drivers (per second of lap time), to see whether improving
     looks like "being a faster driver";
  2. plots style trajectories of the biggest improvers;
  3. for chosen drivers, shows the per-corner style of each lap and where on the track the time was gained.

Usage (from linesight/):
  python scripts/driver_progress.py ../style_vectors/Hockolicious ../replays/tmnf_exchange/Hockolicious/manifest.csv
  python scripts/driver_progress.py ../style_vectors/Hockolicious ../replays/tmnf_exchange/Hockolicious/manifest.csv --drivers tokox17,thomassims
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_driver_styles import FEATURES, braking_corners, lap_features  # noqa: E402

CORNER_PANELS = [
    ("bz_onset_m", "brake point (m before entry; negative = after entry)", "viridis", None),
    ("bz_len_m", "braking zone length (m)", "viridis", None),
    ("l_apex", "line at apex (+1 = inside edge)", "coolwarm", (-0.6, 0.6)),
    ("s", "steering speed (full-lock sweeps / s)", "magma", None),
    ("v_entry_ratio", "entry speed / peak speed", "viridis", (0.4, 1.0)),
    ("corner_time_s", "time in the corner (s)", "viridis", None),
]


def load(style_dir: Path, manifest: Path):
    laps = pd.read_csv(style_dir / "laps.csv")
    corners = pd.read_csv(style_dir / "corners.csv")
    m = pd.read_csv(manifest)[["file", "replay_at", "replay_id"]]
    laps = laps.merge(m, on="file", how="left")
    laps["t"] = laps["lap_time_ms"] / 1000
    laps["date"] = pd.to_datetime(laps["replay_at"], errors="coerce")
    return laps, corners


def within_vs_between(per_lap: pd.DataFrame, min_gain: float):
    """Per feature: change within a driver per second gained, and difference between drivers per second of lap time."""
    rows, pairs = [], []
    for drv, g in per_lap.groupby("driver"):
        if len(g) < 2:
            continue
        slow, fast = g.loc[g["t"].idxmax()], g.loc[g["t"].idxmin()]
        gain = slow["t"] - fast["t"]
        if gain >= min_gain:
            pairs.append(dict(driver=drv, gain=gain, slow_t=slow["t"], fast_t=fast["t"], **{f: (fast[f] - slow[f]) for f in FEATURES}))
    pairs = pd.DataFrame(pairs)
    best = per_lap.loc[per_lap.groupby("driver")["t"].idxmin()]
    rng = np.random.default_rng(0)
    for f in FEATURES:
        sd = best[f].std()
        d = pairs[f] / pairs["gain"]  # change per second gained, one value per driver
        d = d.dropna()
        boot = [rng.choice(d.values, len(d)).mean() for _ in range(2000)]
        ok = best[f].notna()
        slope = np.polyfit(best.loc[ok, "t"], best.loc[ok, f], 1)[0]  # per second of lap time (positive = slower drivers have more)
        rows.append(
            dict(
                feature=f,
                within_per_s_gained=float(d.mean()),
                within_ci_lo=float(np.percentile(boot, 2.5)),
                within_ci_hi=float(np.percentile(boot, 97.5)),
                within_n=int(len(d)),
                between_per_s_faster=float(-slope),
                feature_sd=float(sd),
            )
        )
    return pd.DataFrame(rows), pairs


def plot_slopes(res: pd.DataFrame, n_drivers: int, fig_dir: Path, track: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(res))
    w = 0.38
    within = res["within_per_s_gained"] / res["feature_sd"]
    lo, hi = res["within_ci_lo"] / res["feature_sd"], res["within_ci_hi"] / res["feature_sd"]
    ax.barh(y - w / 2, within, w, xerr=[within - lo, hi - within], color="tab:red", label=f"same driver, slower to faster lap ({n_drivers} drivers)", capsize=2)
    ax.barh(y + w / 2, res["between_per_s_faster"] / res["feature_sd"], w, color="tab:blue", label="different drivers, slower to faster (374 drivers)")
    ax.set_yticks(y)
    ax.set_yticklabels(res["feature"])
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("change in style feature per second of lap time gained, in units of the feature's spread across drivers")
    ax.set_title(f"{track}: does improving look like being a faster driver? (error bars: 95 % bootstrap over drivers)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_progress_slopes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(per_lap: pd.DataFrame, drivers: list, fig_dir: Path, track: str):
    feats = ["brake_onset_m", "brake_len_m", "apex_inside", "line_swing", "steer_speed", "entry_speed"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    cmap = plt.get_cmap("tab10")
    for ax, f in zip(axes.ravel(), feats):
        for i, d in enumerate(drivers):
            g = per_lap[per_lap["driver"] == d].sort_values("date")
            ax.plot(g["t"], g[f], marker="o", color=cmap(i), lw=1.5, label=d)
        ax.set_xlabel("lap time (s), each point = one replay, in date order")
        ax.set_title(f, fontsize=10)
        ax.invert_xaxis()
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{track}: style of the same driver at slower and faster times (right = faster)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_progress_trajectories.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def section_times(profile: dict, bounds: np.ndarray, lap_time: float) -> np.ndarray:
    s, v = profile["s"], np.maximum(profile["speed"], 5.0) / 3.6
    t = np.concatenate([[0], np.cumsum(2.0 / v[:-1])])
    t *= lap_time / t[-1]
    return np.diff(np.interp(np.clip(bounds, s[0], s[-1]), s, t))


def plot_driver(driver: str, laps: pd.DataFrame, corners: pd.DataFrame, corners_def: dict, profiles_dir: Path, fig_dir: Path, track: str):
    g = laps[laps["driver"] == driver].sort_values("t", ascending=False)  # slowest first
    n_c = len(corners_def["corners"])
    c = corners[corners["lap_id"].isin(g["lap_id"])].set_index(["lap_id", "corner"])
    labels = [f"{r.t:.2f}s  {str(r.date)[:10]}" for r in g.itertuples()]

    fig, axes = plt.subplots(len(CORNER_PANELS) + 1, 1, figsize=(max(9, 0.6 * n_c + 5), 2.3 * len(CORNER_PANELS) + 3.5), gridspec_kw=dict(height_ratios=[len(g)] * len(CORNER_PANELS) + [len(CORNER_PANELS) * 0.9]))
    for ax, (col, title, cmap, rng) in zip(axes, CORNER_PANELS):
        mat = np.array([[c[col].get((lid, k), np.nan) for k in range(1, n_c + 1)] for lid in g["lap_id"]], dtype=float)
        lo, hi = rng if rng else (np.nanpercentile(mat, 2), np.nanpercentile(mat, 98))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(range(n_c))
        ax.set_xticklabels(range(1, n_c + 1), fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)

    # where on the track the time was gained: slowest lap minus fastest lap, per corner and per stretch between corners
    cd = corners_def["corners"]
    bounds = np.array([0.0] + [x for c_ in cd for x in (c_["s_start"], c_["s_end"])] + [corners_def["track_length_m"]])
    names = []
    for i in range(len(cd) + 1):
        names += [f"→{i + 1}" if i < len(cd) else "→end"] + ([f"C{i + 1}"] if i < len(cd) else [])
    slow, fast = g.iloc[0], g.iloc[-1]
    ts = section_times(dict(np.load(profiles_dir / f"{slow['lap_id']}.npz")), bounds, slow["t"])
    tf = section_times(dict(np.load(profiles_dir / f"{fast['lap_id']}.npz")), bounds, fast["t"])
    gain = ts - tf
    ax = axes[-1]
    ax.bar(range(len(gain)), gain, color=["tab:red" if n.startswith("C") else "tab:gray" for n in names])
    ax.set_xticks(range(len(gain)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("time gained (s)")
    ax.set_title(f"where {slow['t'] - fast['t']:.2f} s was gained, {slow['t']:.2f}s lap to {fast['t']:.2f}s lap (red = inside a corner, grey = stretch before it)", fontsize=9, loc="left")
    fig.suptitle(f"{track}: {driver}, per-corner style from slowest (top) to fastest (bottom) replay", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_progress_{driver.replace('/', '_')}.png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(dict(section=names, time_gained_s=gain))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track_dir", type=Path, help="style_vectors/<track>")
    ap.add_argument("manifest", type=Path, help="manifest.csv written by fetch_tmx_replays.py (replay dates)")
    ap.add_argument("--drivers", default="", help="comma-separated drivers for the per-corner figures (default: the 3 biggest improvers with 4+ laps)")
    ap.add_argument("--min-gain", type=float, default=0.5, help="seconds between a driver's slowest and fastest lap to count as improvement")
    args = ap.parse_args()

    laps, corners = load(args.track_dir, args.manifest)
    track = args.track_dir.name
    fig_dir = args.track_dir.parent / "figures"
    corners_def = json.load(open(args.track_dir / "corners_def.json"))
    per_lap = lap_features(laps, corners, braking_corners(corners)).reset_index().merge(laps[["lap_id", "date"]], on="lap_id")

    res, pairs = within_vs_between(per_lap, args.min_gain)
    res.round(4).to_csv(args.track_dir / "progress_within_vs_between.csv", index=False)
    pairs.round(3).to_csv(args.track_dir / "progress_pairs.csv", index=False)
    n = len(pairs)
    print(f"{laps['driver'].nunique()} drivers, {(laps.groupby('driver').size() >= 2).sum()} with 2+ laps, {n} improved by >= {args.min_gain} s")
    print(res.assign(z_within=(res.within_per_s_gained / res.feature_sd).round(2), z_between=(res.between_per_s_faster / res.feature_sd).round(2))[["feature", "z_within", "within_n", "z_between"]].to_string(index=False))
    plot_slopes(res, n, fig_dir, track)

    multi = laps.groupby("driver").agg(n=("t", "size"), gain=("t", lambda x: x.max() - x.min())).query("n >= 4").sort_values("gain", ascending=False)
    chosen = [d for d in args.drivers.split(",") if d] or list(multi.index[:3])
    plot_trajectories(per_lap, list(multi.index[:6]), fig_dir, track)
    for d in chosen:
        gain = plot_driver(d, laps, corners, corners_def, args.track_dir / "profiles", fig_dir, track)
        gain.round(3).to_csv(args.track_dir / f"progress_{d.replace('/', '_')}_time_gain.csv", index=False)
        print(f"{d}: {len(laps[laps.driver == d])} laps, {multi.loc[d, 'gain']:.2f} s gained")
    print(f"figures in {fig_dir}")


if __name__ == "__main__":
    main()
