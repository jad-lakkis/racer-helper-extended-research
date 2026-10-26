"""
The driver-style constrained reward (the CMDP of the problem formulation).

For one lap of the agent:

    r_t        = w_p * (progress made on this step) - w_l * |distance to the reference line|          (- w_t * step time, off by default)
    c_style_t  = at the exit of each corner: how far the lap's braking, throttle, line and steering metrics (b, f, l, s)
                 fall outside the driver's estimated ranges; zero on all other steps
    c_fail_t   = 1 on the last step of a lap that failed (not finished, left the route), 0 elsewhere

    adjusted reward   r~_t = r_t - lambda_s * c_style_t - lambda_f * c_fail_t

and after each batch of laps the multipliers move towards the constraints  E[sum_t c_style] <= B  and  E[sum_t c_fail] <= eps:

    lambda_s <- max(0, lambda_s + eta_s * (mean style cost per lap - B))
    lambda_f <- max(0, lambda_f + eta_f * (fraction of failed laps - eps))

The style metrics are computed by trackmania_rl.style_features, the same code that measures replays of human drivers, so the agent is held
to the numbers extracted from the driver's own laps. The ranges are the min and max over the driver's fastest laps, taken per corner.

Only numpy is needed (plus style_features and action_adapter).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackmania_rl.action_adapter import ActionSpec, input_timelines
from trackmania_rl.style_features import (
    Centerline,
    Corner,
    LapSignals,
    PERIOD_MS,
    build_centerline,
    corner_rows,
    project_on_centerline,
)

MAPS_DIR = Path(__file__).resolve().parents[1] / "maps"

STYLE_METRICS = ("b", "f", "l_entry", "l_apex", "l_exit", "s")
# The ranges come from a handful of laps and can be very narrow. A violation is measured in units of max(range width, floor), so that a
# range of width 0.01 does not turn a tiny difference into a huge cost.
WIDTH_FLOORS = {"b": 0.10, "f": 0.10, "l_entry": 0.15, "l_apex": 0.15, "l_exit": 0.15, "s": 1.0}
MAX_COST_PER_METRIC = 3.0  # no single metric of a single corner can cost more than this


# ─────────────────────────────────────────────────────────────────────────────
# The driver
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DriverStyle:
    """Everything about the target driver that the reward needs: the corners and the range of z in each corner."""

    name: str
    track: str
    centerline: Centerline
    corners: List[Corner]
    ranges: Dict[int, Dict[str, Tuple[float, float]]]  # corner idx -> metric -> (lo, hi)
    half_width: float
    approach_m: float = 120.0
    best_lap_time_ms: Optional[int] = None
    lap_times_ms: List[int] = field(default_factory=list)

    @classmethod
    def from_style_vectors(
        cls,
        style_dir: Path,
        driver: str,
        centerline_npy: Optional[Path] = None,
        track_width_m: Optional[float] = None,
        approach_m: float = 120.0,
    ) -> "DriverStyle":
        """
        Build from the output of scripts/extract_style_vectors.py (style_dir = style_vectors/<track>). The centerline and the road width
        are the ones it recorded in corners_def.json unless given (the recorded path is looked up in maps/ by file name if it does not exist here).
        """
        style_dir = Path(style_dir)
        ranges_all = json.load(open(style_dir / "style_ranges.json"))
        if driver not in ranges_all:
            raise KeyError(f"{driver!r} is not in {style_dir / 'style_ranges.json'}; drivers: {sorted(ranges_all)[:10]}...")
        entry = ranges_all[driver]
        defs = json.load(open(style_dir / "corners_def.json"))
        if centerline_npy is None:
            centerline_npy = Path(defs["centerline"])
            if not centerline_npy.exists():
                centerline_npy = MAPS_DIR / centerline_npy.name
        if track_width_m is None:
            track_width_m = float(defs["track_width_m"])
        corners = [Corner(**c) for c in defs["corners"]]
        ranges = {int(k): {m: (v["min"], v["max"]) for m, v in d.items() if m in STYLE_METRICS} for k, d in entry["corners"].items()}
        times = list(entry["lap_times_ms"])
        return cls(
            name=driver,
            track=defs["track"],
            centerline=build_centerline(np.load(centerline_npy)),
            corners=corners,
            ranges=ranges,
            half_width=track_width_m / 2,
            approach_m=approach_m,
            best_lap_time_ms=int(min(times)) if times else None,
            lap_times_ms=times,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cost of one corner, of one lap
# ─────────────────────────────────────────────────────────────────────────────


def corner_style_cost(row: dict, ranges: Dict[str, Tuple[float, float]]) -> Tuple[float, Dict[str, float]]:
    """Sum over the metrics of how far the value is outside [lo, hi], in units of max(hi - lo, floor). Missing values cost nothing."""
    per_metric = {}
    for m in STYLE_METRICS:
        z = row.get(m)
        if m not in ranges or z is None or not np.isfinite(z):
            continue
        lo, hi = ranges[m]
        scale = max(hi - lo, WIDTH_FLOORS[m])
        per_metric[m] = float(min(max(0.0, lo - z, z - hi) / scale, MAX_COST_PER_METRIC))
    return float(sum(per_metric.values())), per_metric


@dataclass
class LapScore:
    base_reward: np.ndarray  # (n,) reward on arriving at each state, index 0 unused
    c_style: np.ndarray  # (n,) the style cost, at the step where each corner is left
    c_fail: np.ndarray  # (n,)
    style_total: float
    failed: bool
    fail_reason: str
    lap_time_ms: Optional[int]
    rows: List[dict]
    per_corner: Dict[int, Dict[str, float]]
    progress_m: float

    def adjusted_reward(self, lambda_s: float, lambda_f: float) -> np.ndarray:
        return self.base_reward - lambda_s * self.c_style - lambda_f * self.c_fail


# ─────────────────────────────────────────────────────────────────────────────
# From the steps of a rollout to a score
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RewardWeights:
    w_progress: float = 0.01  # per metre advanced along the reference line (the existing Linesight reward is 5 / 500)
    w_line: float = 0.001  # per metre of distance to the reference line, per step
    w_time: float = 0.0  # per second of simulated time, per step (0 in the formulation)


class StyleConstraint:
    def __init__(self, driver: DriverStyle, spec: ActionSpec, weights: RewardWeights = RewardWeights(), min_coverage: float = 0.9, max_median_offset_frac: float = 0.6):
        self.driver = driver
        self.spec = spec
        self.weights = weights
        self.min_coverage = min_coverage
        self.max_median_offset_frac = max_median_offset_frac

    def signals(self, positions: np.ndarray, speed_kmh: np.ndarray, actions: np.ndarray, race_time_ms: Optional[int]) -> Tuple[LapSignals, np.ndarray, np.ndarray]:
        """
        positions (n, 3) and speed_kmh (n,) of the n states at decision times 0, dt, 2 dt, ...; actions (n - 1, 3): the executed (steer, throttle, brake).
        race_time_ms is the finishing time if the lap was finished (it replaces the time of the last state), else None.
        """
        n = len(positions)
        dt = self.spec.decision_ms / 1000.0
        t_s = np.arange(n) * dt
        if race_time_ms is not None:
            t_s[-1] = max(race_time_ms / 1000.0, t_s[-2] + 1e-3)
        total_ms = int(round(t_s[-1] * 1000))
        p, e = project_on_centerline(positions, speed_kmh / 3.6, self.driver.centerline, dt)
        brake, accel, u = input_timelines(actions, self.spec)
        n10 = total_ms // PERIOD_MS + 1

        def fit(a):  # the input timelines are one decision per step; make them cover exactly the 10 ms grid of the lap
            if len(a) >= n10:
                return a[:n10]
            return np.concatenate([a, np.repeat(a[-1:], n10 - len(a))])

        sig = LapSignals(t_s=t_s, p=p, e=e, speed_kmh=speed_kmh, brake=fit(brake), accel=fit(accel), u=fit(u), race_time_ms=total_ms)
        return sig, p, e

    def score(self, positions: np.ndarray, speed_kmh: np.ndarray, actions: np.ndarray, finished: bool, race_time_ms: Optional[int] = None) -> LapScore:
        n = len(positions)
        dt = self.spec.decision_ms / 1000.0
        sig, p, e = self.signals(positions, speed_kmh, actions, race_time_ms if finished else None)
        p_mono = np.maximum.accumulate(p)

        w = self.weights
        base = np.zeros(n)
        base[1:] = w.w_progress * np.diff(p_mono) - w.w_line * np.abs(e[1:]) - w.w_time * dt

        d = self.driver
        rows = corner_rows(sig, d.corners, d.half_width, d.approach_m, {})
        c_style = np.zeros(n)
        per_corner = {}
        for c, row in zip(d.corners, rows):
            cost, per_metric = corner_style_cost(row, d.ranges.get(c.idx, {}))
            per_corner[c.idx] = per_metric
            if cost > 0:
                step = int(np.clip(np.ceil(sig.t_at(c.s_end) / dt - 1e-9), 1, n - 1))
                c_style[step] += cost

        coverage = float(p_mono[-1] / d.centerline.s[-1])
        failed, reason = False, ""
        if not finished:
            failed, reason = True, "not finished"
        elif coverage < self.min_coverage:
            failed, reason = True, f"covers only {coverage:.0%} of the reference line"
        elif np.median(np.abs(e)) > self.max_median_offset_frac * d.half_width:
            failed, reason = True, "far from the reference line"
        c_fail = np.zeros(n)
        if failed:
            c_fail[-1] = 1.0
        return LapScore(
            base_reward=base,
            c_style=c_style,
            c_fail=c_fail,
            style_total=float(c_style.sum()),
            failed=failed,
            fail_reason=reason,
            lap_time_ms=int(round(sig.race_time_ms)) if finished else None,
            rows=rows,
            per_corner=per_corner,
            progress_m=float(p_mono[-1]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# The multipliers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Multipliers:
    lambda_s: float = 0.1
    lambda_f: float = 0.5
    eta_s: float = 0.01
    eta_f: float = 0.1
    budget_style: float = 2.0  # B: allowed mean style cost per lap
    budget_fail: float = 0.1  # eps: allowed fraction of failed laps
    max_lambda: float = 100.0

    def update(self, scores: Sequence[LapScore]) -> Dict[str, float]:
        """One dual-ascent step from a batch of laps (the average style cost per lap and the fraction of failed laps)."""
        mean_style = float(np.mean([s.style_total for s in scores]))
        fail_rate = float(np.mean([s.failed for s in scores]))
        self.lambda_s = float(np.clip(self.lambda_s + self.eta_s * (mean_style - self.budget_style), 0.0, self.max_lambda))
        self.lambda_f = float(np.clip(self.lambda_f + self.eta_f * (fail_rate - self.budget_fail), 0.0, self.max_lambda))
        return dict(lambda_s=self.lambda_s, lambda_f=self.lambda_f, mean_style_cost=mean_style, fail_rate=fail_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Picking the output lap
# ─────────────────────────────────────────────────────────────────────────────


def select_target(laps: Sequence[dict], best_time_ms: int, gain_ms: int, style_budget: float) -> Optional[dict]:
    """
    From logged laps (dicts with lap_time_ms, style_total, failed) the fastest valid one with
    lap time <= best_time_ms - gain_ms and style gap <= style_budget; None if no lap qualifies (then the gain is not reachable in this style).
    """
    ok = [lap for lap in laps if not lap["failed"] and lap["lap_time_ms"] is not None and lap["lap_time_ms"] <= best_time_ms - gain_ms and lap["style_total"] <= style_budget]
    return min(ok, key=lambda lap: lap["lap_time_ms"]) if ok else None
