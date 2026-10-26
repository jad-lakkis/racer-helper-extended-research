"""
The driver the agent is asked to drive like, and the context it sees about him.

A DriverPackage holds, for one driver on one track,
    - the style ranges of z = (b, f, l, s) in every corner and the corner definitions (trackmania_rl.style_constraint.DriverStyle),
    - his fastest laps as reference laps: lateral offset and speed along the centerline on a 2 m grid.

At the start of every rollout one reference lap is sampled. During the rollout the agent is told where that lap's line and speed are at
a few look-ahead distances in front of it (contextual RL), and the line term of the reward is the distance to that line.

Packages are small .npz files, built from the output of scripts/extract_style_vectors.py by scripts/build_driver_package.py.
Only numpy is needed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from trackmania_rl.style_constraint import DriverStyle

LOOK_AHEAD_M = (0.0, 15.0, 30.0, 60.0, 100.0, 150.0)
SPEED_CENTER_KMH, SPEED_SCALE_KMH = 150.0, 100.0
N_REFERENCE_FEATURES = 2 * len(LOOK_AHEAD_M) + 1


@dataclass
class ReferenceLap:
    lap_id: str
    lap_time_ms: int
    s: np.ndarray  # (n,) arc length along the centerline, m (uniform grid)
    offset: np.ndarray  # (n,) lateral offset, m (+ = along the centerline normal, the same sign as style_features)
    speed: np.ndarray  # (n,) km/h

    def offset_at(self, p_m):
        """Lateral offset of the reference line at arc length p_m (scalar or array; beyond the ends the end values hold)."""
        return np.interp(p_m, self.s, self.offset)

    def speed_at(self, p_m):
        return np.interp(p_m, self.s, self.speed)


@dataclass
class DriverPackage:
    style: DriverStyle
    refs: List[ReferenceLap]

    # ─── what the agent sees ────────────────────────────────────────────────

    def sample_reference(self, rng: np.random.Generator) -> ReferenceLap:
        return self.refs[int(rng.integers(len(self.refs)))]

    def reference_features(self, ref: ReferenceLap, p_m: float, e_m: float) -> np.ndarray:
        """
        The float features appended to the state, N_REFERENCE_FEATURES numbers of order 1:
        the reference line (offset / half road width) and speed ((km/h - 150) / 100) at p + LOOK_AHEAD_M, then how far the car is from
        the reference line right now (offset difference / half road width). p_m, e_m: the car's arc length and lateral offset on the centerline.
        """
        ahead = p_m + np.asarray(LOOK_AHEAD_M)
        hw = self.style.half_width
        return np.concatenate(
            [ref.offset_at(ahead) / hw, (ref.speed_at(ahead) - SPEED_CENTER_KMH) / SPEED_SCALE_KMH, [(e_m - float(ref.offset_at(p_m))) / hw]]
        ).astype(np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return [f"ref_line_{int(d)}m" for d in LOOK_AHEAD_M] + [f"ref_speed_{int(d)}m" for d in LOOK_AHEAD_M] + ["ref_line_error"]

    # ─── files ──────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        meta = dict(self.style.to_meta(), refs=[dict(lap_id=r.lap_id, lap_time_ms=int(r.lap_time_ms)) for r in self.refs])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(meta)),
            s=self.refs[0].s.astype(np.float32),
            offset=np.stack([r.offset for r in self.refs]).astype(np.float32),
            speed=np.stack([r.speed for r in self.refs]).astype(np.float32),
        )

    @classmethod
    def load(cls, path: Path, centerline_npy: Optional[Path] = None) -> "DriverPackage":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        s = z["s"].astype(np.float64)
        refs = [
            ReferenceLap(r["lap_id"], int(r["lap_time_ms"]), s, z["offset"][i].astype(np.float64), z["speed"][i].astype(np.float64))
            for i, r in enumerate(meta["refs"])
        ]
        return cls(DriverStyle.from_meta(meta, centerline_npy), refs)

    @classmethod
    def build(cls, style_dir: Path, driver: str, k: int = 5, centerline_npy: Optional[Path] = None) -> "DriverPackage":
        """
        From style_vectors/<track>: the driver's k fastest valid laps (the same ones the style ranges are taken from) as reference laps.
        Needs style_vectors/<track>/profiles/<lap_id>.npz, which extract_style_vectors.py writes (they are not committed).
        """
        import pandas as pd

        style_dir = Path(style_dir)
        style = DriverStyle.from_style_vectors(style_dir, driver, centerline_npy)
        laps = pd.read_csv(style_dir / "laps.csv")
        best = laps[laps["driver"] == driver].nsmallest(k, "lap_time_ms")
        if sorted(best["lap_time_ms"].tolist()) != sorted(style.lap_times_ms):
            raise ValueError(
                f"laps.csv and style_ranges.json disagree for {driver!r} (top-k differs: k={k} here, the ranges were made with "
                f"{len(style.lap_times_ms)} laps); pass the same --top-k that extract_style_vectors.py used"
            )
        refs = []
        for _, lap in best.iterrows():
            prof = style_dir / "profiles" / f"{lap['lap_id']}.npz"
            if not prof.exists():
                raise FileNotFoundError(f"{prof} is missing: run scripts/extract_style_vectors.py on the replays again to regenerate the profiles")
            z = np.load(prof)
            refs.append(ReferenceLap(lap["lap_id"], int(lap["lap_time_ms"]), z["s"], z["offset"], z["speed"]))
        if any(len(r.s) != len(refs[0].s) or not np.allclose(r.s, refs[0].s) for r in refs):
            raise ValueError("profiles are not on the same distance grid")
        return cls(style, refs)

    def describe(self) -> str:
        times = ", ".join(f"{r.lap_time_ms / 1000:.2f}" for r in self.refs)
        return f"{self.style.name} on {self.style.track}: {len(self.refs)} reference laps ({times} s), {len(self.style.corners)} corners"


def reference_feature_mean_std() -> tuple:
    """Mean and std to normalise the reference features with (they are already of order 1, so identity)."""
    return np.zeros(N_REFERENCE_FEATURES, dtype=np.float32), np.ones(N_REFERENCE_FEATURES, dtype=np.float32)
