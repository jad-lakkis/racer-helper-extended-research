#!/usr/bin/env python3
"""
Group drivers by driving style, using the per-corner style vectors from extract_style_vectors.py.

1. Each driver becomes one compact style vector (mean over their laps, see FEATURES).
2. Lap time and steering device (pad / wheel vs keyboard) explain a lot of the raw variation, so they are
   regressed out. What is left is style relative to what a driver of that speed would normally do.
   (Clustering the raw vectors mostly recovers speed tiers; --raw does that for reference.)
3. Gaussian mixtures with 1..8 components and three covariance shapes are fitted and compared by BIC.
   Hard k-means with the silhouette score is shown next to it, and compared with the same statistic on
   column-shuffled data, which has the same marginals but no structure.
4. Stability: refit on random 80 % subsets and compare labels (adjusted Rand index).
5. For each --me driver: nearest drivers in style space that are faster, with the raw feature differences.
   Those are the style-matched targets for a faster lap.

Usage (from linesight/):
  python scripts/cluster_driver_styles.py ../style_vectors --me jadlakkisjad
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_style_vectors import driver_matrix, plot_group_corner_profiles, plot_me_vs_groups  # noqa: E402

warnings.filterwarnings("ignore")

# name -> (description, unit)
FEATURES = {
    "brake_share": ("share of braking-corner time spent braking", "fraction"),
    "brake_onset_m": ("brake point before braking corners", "m"),
    "throttle": ("full-throttle share of corner time", "fraction"),
    "coast": ("share of corner time with neither pedal", "fraction"),
    "apex_inside": ("line at the apex (+1 = inside edge)", "road half-widths"),
    "line_swing": ("apex minus mean of entry and exit line", "road half-widths"),
    "steer_speed": ("steering speed", "full-lock sweeps / s"),
    "steer_reversals": ("left-right reversals per corner", "count"),
    "entry_speed": ("corner entry speed / peak speed", "ratio"),
}
CONDITIONING = ["braking_aggression", "risk_tolerance_proxy", "oversteer_understeer_score", "corner_entry_speed_ratio"]
K_RANGE = range(1, 9)
COV_TYPES = ("spherical", "diag", "full")


def braking_corners(corners: pd.DataFrame) -> list:
    """Corners where a typical lap spends time braking in or just before the corner."""
    m = corners.groupby("corner")[["b", "b_approach"]].median().sum(axis=1)
    return [int(c) for c in m.index[m > 0.15]]


def compact_features(laps: pd.DataFrame, corners: pd.DataFrame, brake_corners: list) -> pd.DataFrame:
    c = corners[corners["lap_id"].isin(laps["lap_id"])].copy()
    c["swing"] = c["l_apex"] - (c["l_entry"] + c["l_exit"]) / 2
    br = c[c["corner"].isin(brake_corners)]
    per_lap = pd.DataFrame(
        {
            "brake_share": br.groupby("lap_id")["b"].mean(),
            "brake_onset_m": br.groupby("lap_id")["brake_onset_m"].mean(),
            "throttle": c.groupby("lap_id")["f"].mean(),
            "coast": c.groupby("lap_id")["coast"].mean(),
            "apex_inside": c.groupby("lap_id")["l_apex"].mean(),
            "line_swing": c.groupby("lap_id")["swing"].mean(),
            "steer_speed": c.groupby("lap_id")["s"].mean(),
            "steer_reversals": c.groupby("lap_id")["steer_reversals"].mean(),
            "entry_speed": c.groupby("lap_id")["v_entry_ratio"].mean(),
        }
    )
    per_lap = per_lap.join(laps.set_index("lap_id")[["driver", "lap_time_ms", "steer_input"]])
    per_lap["analog"] = (per_lap["steer_input"] == "analog").astype(float)
    per_lap["t"] = per_lap["lap_time_ms"] / 1000
    g = per_lap.groupby("driver")
    out = g[list(FEATURES)].mean()
    out["n_laps"] = g.size()
    out["best_time_s"] = g["t"].min()
    out["mean_time_s"] = g["t"].mean()
    out["analog"] = g["analog"].mean()
    return out


def split_half_reliability(laps: pd.DataFrame, corners: pd.DataFrame, brake_corners: list, seed: int = 0) -> dict:
    """Correlation between two random halves of each driver's laps (Spearman-Brown corrected), drivers with >= 2 laps."""
    rng = np.random.default_rng(seed)
    multi = laps.groupby("driver").filter(lambda g: len(g) >= 2)
    halves = {}
    for drv, g in multi.groupby("driver"):
        ids = g["lap_id"].values.copy()
        rng.shuffle(ids)
        halves[drv] = (ids[: len(ids) // 2], ids[len(ids) // 2 :])
    out = {}
    cols = {}
    for h in (0, 1):
        sub = []
        for drv, ids in ((d, hh[h]) for d, hh in halves.items()):
            sl = laps[laps["lap_id"].isin(ids)].assign(driver=drv)
            sub.append(compact_features(sl, corners, brake_corners)[list(FEATURES)].rename(index={drv: drv}))
        cols[h] = pd.concat(sub)
    for f in FEATURES:
        r = np.corrcoef(cols[0][f], cols[1][f])[0, 1]
        out[f] = float(2 * r / (1 + r))
    return out


def adjust_for_skill(X: pd.DataFrame, info: pd.DataFrame):
    """Residuals of each feature after regressing on log(mean lap time) and the share of analog laps."""
    Z = np.column_stack([np.ones(len(X)), np.log(info["mean_time_s"]), info["analog"]])
    B = np.linalg.lstsq(Z, X.values, rcond=None)[0]
    res = X.values - Z @ B
    r2 = 1 - res.var(0) / X.values.var(0)
    return pd.DataFrame(res, index=X.index, columns=X.columns), dict(zip(X.columns, r2.round(3)))


def fit_gmms(Z: np.ndarray) -> pd.DataFrame:
    rows = []
    for cov in COV_TYPES:
        for k in K_RANGE:
            try:
                g = GaussianMixture(k, covariance_type=cov, n_init=5, reg_covar=1e-3, random_state=0).fit(Z)
                rows.append(dict(cov=cov, k=k, bic=g.bic(Z)))
            except ValueError:
                pass
    return pd.DataFrame(rows)


def silhouettes(Z: np.ndarray, n_null: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs, null = {}, {k: [] for k in range(2, 9)}
    for k in range(2, 9):
        obs[k] = silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z))
    for _ in range(n_null):
        Zs = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        for k in range(2, 9):
            null[k].append(silhouette_score(Zs, KMeans(k, n_init=3, random_state=0).fit_predict(Zs)))
    return obs, {k: np.array(v) for k, v in null.items()}


def stability(Z: np.ndarray, k: int, cov: str, n_boot: int = 30, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    full = GaussianMixture(k, covariance_type=cov, n_init=5, reg_covar=1e-3, random_state=0).fit(Z).predict(Z)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(len(Z), int(0.8 * len(Z)), replace=False)
        sub = GaussianMixture(k, covariance_type=cov, n_init=3, reg_covar=1e-3, random_state=0).fit(Z[idx]).predict(Z[idx])
        aris.append(adjusted_rand_score(full[idx], sub))
    return float(np.mean(aris))


def plot_model_selection(bic: pd.DataFrame, obs: dict, null: dict, fig_dir: Path, track: str, tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for cov, g in bic.groupby("cov"):
        axes[0].plot(g["k"], g["bic"], marker="o", label=cov)
    axes[0].set_xlabel("number of clusters k")
    axes[0].set_ylabel("BIC (lower is better)")
    axes[0].set_title("Gaussian mixture: BIC by covariance shape")
    axes[0].legend()
    ks = list(obs)
    axes[1].plot(ks, [obs[k] for k in ks], marker="o", color="k", label="observed")
    axes[1].fill_between(ks, [np.percentile(null[k], 5) for k in ks], [np.percentile(null[k], 95) for k in ks], color="0.8", label="column-shuffled data (5-95 %)")
    axes[1].set_xlabel("number of clusters k")
    axes[1].set_ylabel("k-means silhouette")
    axes[1].set_title("Is there more structure than in shuffled data?")
    axes[1].legend()
    fig.suptitle(f"{track}: model selection ({tag})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_style_model_selection_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_space(info: pd.DataFrame, Z: np.ndarray, labels: np.ndarray, fig_dir: Path, track: str, me: list, tag: str):
    pca = PCA(n_components=2).fit(Z)
    xy = pca.transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.get_cmap("tab10")
    size = 14 + 5 * np.minimum(info["n_laps"].values, 8)
    for k in sorted(set(labels)):
        m = labels == k
        axes[0].scatter(xy[m, 0], xy[m, 1], s=size[m], color=cmap(int(k) % 10), edgecolor="k", lw=0.3, alpha=0.85, label=f"cluster {k} (n={m.sum()})")
    sc = axes[1].scatter(xy[:, 0], xy[:, 1], s=size, c=info["best_time_s"], cmap="viridis_r", edgecolor="k", lw=0.3)
    fig.colorbar(sc, ax=axes[1], label="driver's best lap (s)")
    for ax in axes:
        for name in me:
            if name in info.index:
                i = info.index.get_loc(name)
                ax.scatter(xy[i, 0], xy[i, 1], marker="*", s=420, color="gold", edgecolor="k", zorder=5)
                ax.annotate(name, xy[i], xytext=(8, 8), textcoords="offset points", fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    axes[0].legend(fontsize=8)
    axes[0].set_title("drivers in style space, coloured by cluster (size = laps)")
    axes[1].set_title("same plot coloured by speed")
    fig.suptitle(f"{track}: driver styles ({tag})")
    fig.savefig(fig_dir / f"{track}_style_space_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def style_axes(Z: np.ndarray, n: int = 3):
    pca = PCA(n_components=n).fit(Z)
    return pca, pd.DataFrame(pca.components_.T, index=list(FEATURES), columns=[f"PC{i + 1}" for i in range(n)])


def plot_axes(pca: PCA, loadings: pd.DataFrame, fig_dir: Path, track: str):
    fig, axes = plt.subplots(1, loadings.shape[1], figsize=(4.2 * loadings.shape[1], 4.6), sharey=True)
    for ax, pc in zip(axes, loadings.columns):
        v = loadings[pc].values
        ax.barh(range(len(v)), v, color=["tab:red" if x > 0 else "tab:blue" for x in v])
        ax.set_yticks(range(len(v)))
        ax.set_yticklabels(loadings.index, fontsize=8)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{pc} ({pca.explained_variance_ratio_[int(pc[2:]) - 1]:.0%} of variance)", fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle(f"{track}: main style axes (loadings on standardised, skill-adjusted features)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_style_axes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cards(info: pd.DataFrame, Zres: pd.DataFrame, labels: np.ndarray, fig_dir: Path, track: str, me: list):
    ks = sorted(set(labels))
    z = (Zres - Zres.mean()) / Zres.std(ddof=0)
    fig, axes = plt.subplots(1, len(ks), figsize=(3.4 * len(ks) + 2, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, ks):
        m = labels == k
        v = z[m].mean().values
        ax.barh(range(len(FEATURES)), v, color=["tab:red" if x > 0 else "tab:blue" for x in v])
        for name in me:
            if name in z.index:
                ax.scatter(z.loc[name].values, range(len(FEATURES)), marker="*", s=70, color="gold", edgecolor="k", zorder=5, label=name)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(range(len(FEATURES)))
        ax.set_yticklabels(list(FEATURES), fontsize=8)
        ax.set_xlim(-2.5, 2.5)
        ax.set_title(f"cluster {k}: {m.sum()} drivers\nmedian best {info.loc[m, 'best_time_s'].median():.2f}s", fontsize=9)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.supxlabel("z-score of style relative to drivers of the same speed and steering device")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{track}_style_cluster_cards.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_track(track_dir: Path, fig_dir: Path, me: list, raw: bool):
    track = track_dir.name
    laps = pd.read_csv(track_dir / "laps.csv")
    corners = pd.read_csv(track_dir / "corners.csv")
    bc = braking_corners(corners)
    print(f"\n== {track}: {laps['driver'].nunique()} drivers; braking corners {bc}")

    X = compact_features(laps, corners, bc)
    feat = X[list(FEATURES)]
    X = X.assign(**{f: feat[f].fillna(feat[f].median()) for f in FEATURES})
    rel = split_half_reliability(laps, corners, bc)
    print("split-half reliability (drivers with >= 2 laps):", {k: round(v, 2) for k, v in rel.items()})

    res, r2 = adjust_for_skill(X[list(FEATURES)], X)
    print("share of each feature explained by lap time + steering device:", r2)

    out = dict(track=track, braking_corners=bc, split_half_reliability=rel, r2_lap_time_and_device=r2, n_drivers=len(X))
    for tag, M in ((("raw", X[list(FEATURES)]),) if raw else ()) + (("adjusted", res),):
        Z = StandardScaler().fit_transform(M)
        bic = fit_gmms(Z)
        best = bic.loc[bic["bic"].idxmin()]
        k, cov = int(best["k"]), best["cov"]
        obs, null = silhouettes(Z)
        plot_model_selection(bic, obs, null, fig_dir, track, tag)
        sil_ok = {kk: bool(obs[kk] > np.percentile(null[kk], 95)) for kk in obs}
        print(f"\n[{tag}] best BIC: k={k} ({cov}); BIC by k (best cov each): "
              + str({int(kk): int(v) for kk, v in bic.groupby("k")["bic"].min().items()}))
        print(f"[{tag}] k-means silhouette observed vs shuffled-95%: "
              + str({kk: (round(obs[kk], 3), round(float(np.percentile(null[kk], 95)), 3)) for kk in obs}))
        entry = dict(best_bic_k=k, best_bic_cov=cov, silhouette=obs, silhouette_null95={kk: float(np.percentile(null[kk], 95)) for kk in null}, silhouette_above_null=sil_ok)
        if tag == "adjusted":
            kk = k if k >= 2 else 3
            g = GaussianMixture(kk, covariance_type=cov if k >= 2 else "diag", n_init=10, reg_covar=1e-3, random_state=0).fit(Z)
            post = g.predict_proba(Z)
            labels = post.argmax(1)
            order = pd.Series(X["best_time_s"].values).groupby(labels).median().sort_values().index  # cluster 0 = fastest group
            remap = {old: new for new, old in enumerate(order)}
            labels = np.array([remap[v] for v in labels])
            post = post[:, list(order)]
            entry.update(
                k_used=int(kk),
                k_chosen_by_bic=bool(k >= 2),
                stability_ari=stability(Z, kk, cov if k >= 2 else "diag"),
                mean_max_posterior=float(post.max(1).mean()),
            )
            print(f"[adjusted] using k={kk}; stability ARI {entry['stability_ari']:.2f}; mean max posterior {entry['mean_max_posterior']:.2f}")
            plot_space(X, Z, labels, fig_dir, track, me, tag)
            pca3, loadings = style_axes(Z)
            plot_axes(pca3, loadings, fig_dir, track)
            entry["style_axes"] = dict(explained_variance=pca3.explained_variance_ratio_.round(3).tolist(), loadings=loadings.round(3).to_dict())
            plot_cards(X, res, labels, fig_dir, track, me)

            drv = X.assign(cluster=labels)
            for c in range(kk):
                drv[f"p_cluster_{c}"] = post[:, c]
            drv.join(res.add_prefix("resid_")).to_csv(track_dir / "driver_style_clusters.csv")

            wide = driver_matrix(laps, corners)
            groups = {f"cluster {c}": list(drv.index[drv["cluster"] == c]) for c in range(kk)}
            plot_group_corner_profiles(groups, wide, fig_dir, track, "style_cluster", me)
            cl = {}
            for c, members in groups.items():
                lp = laps[laps["driver"].isin(members)]
                cl[c] = dict(
                    n_drivers=len(members),
                    median_best_time_s=float(drv.loc[members, "best_time_s"].median()),
                    mean_features={f: float(drv.loc[members, f].mean()) for f in FEATURES},
                    style_vs_peers_z={f: float(((res.loc[members, f].mean()) / res[f].std(ddof=0))) for f in FEATURES},
                    rl_conditioning_mean=lp[CONDITIONING].mean().round(3).to_dict(),
                )
            entry["clusters"] = cl
            max_z = max(abs(v) for c in cl.values() for v in c["style_vs_peers_z"].values())
            entry["clusters_distinct"] = bool(max_z > 0.5)
            print(f"[adjusted] largest cluster-mean z on any feature: {max_z:.2f} "
                  + ("(clusters differ in style)" if max_z > 0.5 else "(clusters differ in spread, not in mean style: no distinct style groups)"))

            for m in [m for m in me if m in X.index]:
                plot_me_vs_groups(groups, wide, fig_dir, track, "style_cluster", m)
                i = X.index.get_loc(m)
                d = np.linalg.norm(Z - Z[i], axis=1)
                cand = X.assign(distance=d).drop(index=m)
                faster = cand[cand["best_time_s"] < X.loc[m, "best_time_s"] - 0.3].nsmallest(10, "distance")
                diff = (X.loc[faster.index, list(FEATURES)] - X.loc[m, list(FEATURES)]).add_prefix("diff_")
                tbl = faster[["best_time_s", "n_laps", "distance"]].join(diff)
                tbl.insert(0, "cluster", drv.loc[faster.index, "cluster"])
                tbl.to_csv(track_dir / f"style_neighbours_{m}.csv")
                entry[f"{m}"] = dict(
                    cluster=int(drv.loc[m, "cluster"]),
                    cluster_probabilities=[round(float(p), 3) for p in post[i]],
                    features=X.loc[m, list(FEATURES)].round(3).to_dict(),
                    style_vs_peers_z=(res.loc[m] / res.std(ddof=0)).round(2).to_dict(),
                    faster_style_neighbours=tbl.round(3).reset_index().to_dict("records")[:5],
                )
                print(f"\n{m}: best {X.loc[m, 'best_time_s']:.2f}s, cluster {int(drv.loc[m, 'cluster'])}, probabilities {np.round(post[i], 2)}")
                print("style relative to drivers of the same speed (z):", (res.loc[m] / res.std(ddof=0)).round(2).to_dict())
                print("closest faster drivers in style space:\n" + tbl.round(2).to_string())
        out[tag] = entry
    json.dump(out, open(track_dir / "style_clusters.json", "w"), indent=2, default=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("style_dir", type=Path)
    ap.add_argument("--me", default="jadlakkisjad", help="comma-separated driver login(s)")
    ap.add_argument("--raw", action="store_true", help="also model the unadjusted vectors, for reference")
    ap.add_argument("--min-drivers", type=int, default=60)
    args = ap.parse_args()
    fig_dir = args.style_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    me = [m for m in args.me.split(",") if m]
    for d in sorted(args.style_dir.iterdir()):
        if (d / "laps.csv").exists() and pd.read_csv(d / "laps.csv")["driver"].nunique() >= args.min_drivers:
            run_track(d, fig_dir, me, args.raw)
        elif (d / "laps.csv").exists():
            print(f"skip {d.name}: fewer than {args.min_drivers} drivers")


if __name__ == "__main__":
    main()
