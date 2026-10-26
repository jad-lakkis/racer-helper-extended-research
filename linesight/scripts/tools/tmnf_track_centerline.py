#!/usr/bin/env python3
"""
A centerline (x, y, z every 0.5 m, same format as linesight/maps/*.npy) from the first lap of a replay, for tracks that have no
centerline file. The line is the replay's own driven line, smoothed with a spline, so offsets measured against it are relative to
that driver's line, not to the middle of the road.

  python scripts/tools/tmnf_track_centerline.py some.Replay.Gbx --out ../tracks/MyTrack/centerline.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_driver_profile import _load_ghost  # noqa: E402
from extract_style_vectors import first_lap_end  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--noise-m", type=float, default=1.0, help="position noise assumed by the smoothing spline")
    args = ap.parse_args()

    g, _, rt, n = _load_ghost(str(args.replay))
    P = np.array([[r.position.x, r.position.y, r.position.z] for r in g.records[: n + 1]])
    end = first_lap_end(P[:, [0, 2]])
    if end is not None:
        P = P[: end + 1]
    # drop the standing start (the first samples barely move) so that the line starts when the car does
    moving = np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1) > 0.2
    P = P[int(np.argmax(moving)) :]
    tck, _ = splprep([P[:, 0], P[:, 1], P[:, 2]], s=len(P) * args.noise_m**2, k=3)
    u = np.linspace(0, 1, 20000)
    x, y, z = splev(u, tck)
    line = np.column_stack([x, y, z])
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1))])
    t = np.arange(0, arc[-1], 0.5)
    out = np.column_stack([np.interp(t, arc, line[:, i]) for i in range(3)]).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f"{len(out)} points, {arc[-1]:.0f} m, from the first lap of {args.replay.name}")


if __name__ == "__main__":
    main()
