#!/usr/bin/env python3
"""
Centerline, width and elevation of a Trackmania 2020 map whose road is built from embedded custom items.

Step 1 (C#, once per map):  dotnet run --project scripts/tools/tm2020_map_dump -- <map.Map.Gbx> <dump dir>
Step 2 (this script):       python scripts/tools/tm2020_centerline.py <dump dir> --out <out dir> [--start-item NAME]

The road surface is the mesh part with the given material (default CustomPlastic, the asphalt of this map; the
table printed at the start shows every material with its cross-section width so another map can pick its own).
Its vertices are placed in the map (world = R_yaw (local + pivot) + position, which makes the pieces join exactly),
rasterised at 1 m, thinned to a skeleton and walked as one loop. The skeleton sits on a 1 m grid, so each point is then moved
to the middle of the road vertices across it (which also gives the width), and the loop is smoothed and resampled every 0.5 m.

Writes to <out dir>:  centerline.npy (n, 3: x, y, z, 0.5 m apart, same format as linesight/maps/*.npy),
                      width.npy (n,), summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

RES = 1.0  # raster resolution in metres
DILATE = 3  # pixels, joins neighbouring vertices into a solid strip before thinning
STEP = 0.5  # centerline spacing in metres


def place(P: np.ndarray, p: dict) -> np.ndarray:
    th = p["yaw_pitch_roll"][0]
    c, s = np.cos(th), np.sin(th)
    L = P + np.array(p["pivot"])
    return np.stack([L[:, 0] * c + L[:, 2] * s + p["pos"][0], L[:, 1] + p["pos"][1], -L[:, 0] * s + L[:, 2] * c + p["pos"][2]], 1)


def norm(item: str) -> str:
    """Item ids in placements.json and in the extracted file names differ by slash style and a leading Items/."""
    item = item.replace("\\", "/")
    return item[len("Items/"):] if item.lower().startswith("items/") else item


def load_vertices(dump: Path, material: str, min_vertices: int):
    meta = json.load(open(dump / "meshes.json"))
    placements = {norm(p["item"]): p for p in json.load(open(dump / "placements.json"))}
    print("materials (parts, vertices):")
    mats = {}
    for m in meta:
        k = m["link"].split("\\")[-1] or "(none)"
        a = mats.setdefault(k, [0, 0])
        a[0] += 1
        a[1] += m["vertices"]
    for k, (n, v) in sorted(mats.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:28s} {n:3d} parts {v:7d} vertices")
    pts, used = [], []
    for m in meta:
        if not m["link"].endswith(material) or m["vertices"] < min_vertices or norm(m["item"]) not in placements:
            continue
        P = np.fromfile(dump / (m["file"] + ".pos.f32"), "<f4").reshape(-1, 3)
        pts.append(place(P, placements[norm(m["item"])]))
        used.append(m["item"])
    return np.vstack(pts), sorted(set(used)), placements


def walk_loop(sk: np.ndarray) -> np.ndarray:
    K = np.ones((3, 3), int)
    K[1, 1] = 0
    for _ in range(60):  # prune short spurs
        ends = (ndi.convolve(sk.astype(int), K, mode="constant") * sk <= 1) & sk
        if not ends.any():
            break
        sk[ends] = False
    pix = np.argwhere(sk)
    idx = {tuple(p): i for i, p in enumerate(pix)}
    nbrs = [[idx[(r + dr, c + dc)] for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (r + dr, c + dc) in idx] for r, c in pix]
    cur, order, seen = 0, [0], {0}
    while True:
        cand = [n for n in nbrs[cur] if n not in seen]
        if not cand:
            break
        cand.sort(key=lambda n: abs(pix[n][0] - pix[cur][0]) + abs(pix[n][1] - pix[cur][1]))
        cur = cand[0]
        order.append(cur)
        seen.add(cur)
    if len(order) < 0.98 * len(pix):
        raise SystemExit(f"skeleton is not a single loop: walked {len(order)} of {len(pix)} pixels")
    return pix[order]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--material", default="CustomPlastic", help="material link suffix of the road surface")
    ap.add_argument("--min-vertices", type=int, default=1000, help="ignore small parts of the road material (e.g. gantries)")
    ap.add_argument("--start-item", help="substring of the item id that marks the start line; the loop starts at its position")
    ap.add_argument("--reverse", action="store_true", help="drive the loop the other way round")
    ap.add_argument("--smooth-m", type=float, default=4.0, help="gaussian smoothing of the centerline, in metres")
    args = ap.parse_args()

    V, used, placements = load_vertices(args.dump, args.material, args.min_vertices)
    print(f"road: {len(V)} vertices from {len(used)} items: {used}")
    x0, z0 = V[:, 0].min() - 10, V[:, 2].min() - 10
    nx, nz = int((V[:, 0].max() + 10 - x0) / RES) + 1, int((V[:, 2].max() + 10 - z0) / RES) + 1
    mask = np.zeros((nz, nx), bool)
    mask[((V[:, 2] - z0) / RES).astype(int), ((V[:, 0] - x0) / RES).astype(int)] = True
    mask = ndi.binary_closing(ndi.binary_dilation(mask, iterations=DILATE), iterations=3)
    loop = walk_loop(skeletonize(mask))
    xz = np.stack([loop[:, 1] * RES + x0, loop[:, 0] * RES + z0], 1)
    if np.linalg.norm(xz[0] - xz[-1]) > 5:
        raise SystemExit("loop does not close")

    if args.start_item:
        name = next(k for k in placements if args.start_item in k)  # keys are already normalised
        sp = placements[name]["pos"]
        xz = np.roll(xz, -int(np.argmin(np.linalg.norm(xz - np.array([sp[0], sp[2]]), axis=1))), axis=0)
    if args.reverse:
        xz = np.roll(xz[::-1], 1, axis=0)

    # resample the closed polyline at constant spacing, smooth it, resample again
    def resample(p, step):
        q = np.vstack([p, p[:1]])
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
        t = np.arange(0, s[-1], step)
        return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)

    def smooth(p, sigma):
        return np.stack([gaussian_filter1d(p[:, 0], sigma, mode="wrap"), gaussian_filter1d(p[:, 1], sigma, mode="wrap")], 1)

    p = smooth(resample(xz, 1.0), args.smooth_m)  # one point per metre while refining
    tree = cKDTree(V[:, [0, 2]])
    width = np.zeros(len(p))
    for _ in range(4):
        # The skeleton sits on a 1 m grid. Move each point to the middle of the road vertices across it,
        # measured along the normal in a short window along the track.
        t = np.roll(p, -2, axis=0) - np.roll(p, 2, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        off = np.zeros(len(p))
        for i, nb in enumerate(tree.query_ball_point(p, 14.0)):
            d = V[nb][:, [0, 2]] - p[i]
            along, lat = d @ t[i], d @ n[i]
            m = np.abs(along) < 3.0
            if m.sum() >= 4:
                off[i] = (lat[m].min() + lat[m].max()) / 2
                width[i] = lat[m].max() - lat[m].min()
        p = smooth(p + n * gaussian_filter1d(off, 3.0, mode="wrap")[:, None], 3.0)
    width = gaussian_filter1d(width, 5.0, mode="wrap")
    q = np.vstack([p, p[:1]])
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    new_arc = np.arange(0, arc[-1], STEP)
    width = np.interp(new_arc, arc, np.append(width, width[0]))
    p = np.stack([np.interp(new_arc, arc, q[:, 0]), np.interp(new_arc, arc, q[:, 1])], 1)

    y = np.array([np.median(V[i, 1]) for i in tree.query_ball_point(p, 6.0)])
    y = gaussian_filter1d(y, 10.0, mode="wrap")  # 5 m; the median still picks up kerb and edge vertices
    cl = np.column_stack([p[:, 0], y, p[:, 1]]).astype(np.float32)
    length = float(np.sum(np.linalg.norm(np.diff(np.vstack([cl, cl[:1]])[:, [0, 2]], axis=0), axis=1)))
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "centerline.npy", cl)
    np.save(args.out / "width.npy", width.astype(np.float32))
    summary = dict(
        points=len(cl),
        lap_length_m=round(length, 1),
        width_m=dict(median=round(float(np.median(width)), 1), p5=round(float(np.percentile(width, 5)), 1), p95=round(float(np.percentile(width, 95)), 1)),
        elevation_m=dict(min=round(float(y.min()), 1), max=round(float(y.max()), 1), range=round(float(y.max() - y.min()), 1)),
        road_items=used,
        start=[round(float(cl[0, 0]), 1), round(float(cl[0, 2]), 1)],
    )
    json.dump(summary, open(args.out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
