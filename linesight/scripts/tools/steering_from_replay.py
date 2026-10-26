#!/usr/bin/env python3
"""
Estimate the steering the car actually received from the path it drove, for any replay (keyboard or analog).

A replay stores the driver's inputs and the car's position every 100 ms, not the steering angle of the wheels. For keyboard drivers
the inputs are only left / straight / right, and the game smooths them into a real steering angle that is not saved.
What can be recovered is the effective steering: how much steering the path implies at that speed.

Model: u(t) ~ g(speed) * path_curvature(t + LAG), fitted on replays with analog steering (where the true value is stored) and clipped to [-1, 1].
Positive u = right, the same for keyboard and analog replays.

  fit:      python scripts/tools/steering_from_replay.py fit ../style_vectors/Hockolicious/laps.csv ../replays --out scripts/tools/steering_model.json
  estimate: python scripts/tools/steering_from_replay.py estimate some.Replay.Gbx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_driver_profile import _load_ghost  # noqa: E402
from extract_style_vectors import _steer_signal  # noqa: E402

LAG = 2  # steps of 100 ms between the steering and its effect on the path
SPEED_BINS = [0, 25, 45, 60, 75, 90, 200]  # m/s
MODEL = Path(__file__).with_name("steering_model.json")


def kinematics(P: np.ndarray):
    """Speed (m/s) and path curvature (1/m) every 100 ms from (x, z) positions."""
    v = np.zeros_like(P)
    v[1:-1] = (P[2:] - P[:-2]) / 0.2
    v[0], v[-1] = v[1], v[-2]
    sp = np.linalg.norm(v, axis=1)
    h = gaussian_filter1d(np.unwrap(np.arctan2(v[:, 1], v[:, 0])), 1.0)
    return sp, np.gradient(h, 0.1) / np.maximum(sp, 5.0)


def load_lap(path):
    g, rs, rt, n = _load_ghost(str(path))
    P = np.array([[r.position.x, r.position.z] for r in g.records[: n + 1]])
    u10, kind = _steer_signal(g.control_entries, rs, int(rt // 10) + 1)
    u = np.array([u10[i * 10 : (i + 1) * 10].mean() for i in range(len(P) - 1)] + [u10[-1]])  # mean input over each 100 ms step
    sp, kappa = kinematics(P)
    return u, sp, kappa, kind


def estimate(sp: np.ndarray, kappa: np.ndarray, model: dict) -> np.ndarray:
    k = np.roll(kappa, -LAG)  # steering at step i shows up in the path LAG steps later
    k[-LAG:] = 0
    out = np.zeros_like(k)
    for (lo, hi), g in zip(zip(model["bins"], model["bins"][1:]), model["slope"]):
        m = (sp >= lo) & (sp < hi)
        out[m] = g * k[m]
    return np.clip(out, -1, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("laps_csv", type=Path)
    f.add_argument("replays", type=Path)
    f.add_argument("--out", type=Path, default=MODEL)
    f.add_argument("--n-laps", type=int, default=120)
    e = sub.add_parser("estimate")
    e.add_argument("replay", type=Path)
    args = ap.parse_args()

    if args.cmd == "fit":
        import pandas as pd

        laps = pd.read_csv(args.laps_csv)
        files = {p.name: p for p in args.replays.rglob("*.Replay.Gbx")}
        names = [n for n in laps[laps.steer_input == "analog"]["file"] if n in files][: args.n_laps]
        data = []
        for n in names:
            try:
                u, sp, kappa, kind = load_lap(files[n])
            except SystemExit:
                continue
            data.append((u[: len(u) - LAG][5:-5], sp[LAG:][5:-5], kappa[LAG:][5:-5]))
        split = int(len(data) * 2 / 3)
        cat = lambda ds: [np.concatenate([d[i] for d in ds]) for i in range(3)]  # noqa: E731
        (ut, vt, kt), (ue, ve, ke) = cat(data[:split]), cat(data[split:])
        slope = []
        for lo, hi in zip(SPEED_BINS, SPEED_BINS[1:]):
            m = (vt >= lo) & (vt < hi)
            slope.append(float((ut[m] * kt[m]).sum() / max((kt[m] ** 2).sum(), 1e-9)) if m.sum() > 50 else 0.0)
        model = dict(bins=SPEED_BINS, slope=slope, lag_steps=LAG, fitted_on_laps=split, tested_on_laps=len(data) - split)
        pred = np.zeros_like(ue)
        for (lo, hi), g in zip(zip(SPEED_BINS, SPEED_BINS[1:]), slope):
            m = (ve >= lo) & (ve < hi)
            pred[m] = np.clip(g * ke[m], -1, 1)
        ok = ve >= 25
        model["heldout"] = dict(
            corr=float(np.corrcoef(ue[ok], pred[ok])[0, 1]),
            r2=float(1 - ((ue[ok] - pred[ok]) ** 2).sum() / ((ue[ok] - ue[ok].mean()) ** 2).sum()),
            rmse=float(np.sqrt(((ue[ok] - pred[ok]) ** 2).mean())),
        )
        json.dump(model, open(args.out, "w"), indent=2)
        print(json.dumps(model, indent=2))
    else:
        model = json.load(open(MODEL))
        u, sp, kappa, kind = load_lap(args.replay)
        est = estimate(sp, kappa, model)
        print(f"{kind} steering; estimated vs recorded input correlation {np.corrcoef(u[5:-5], est[5:-5])[0, 1]:.2f}")


if __name__ == "__main__":
    main()
