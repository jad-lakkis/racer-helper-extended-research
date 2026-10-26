"""
Per-corner driver style features, shared by the replay extractor (scripts/extract_style_vectors.py) and the reinforcement-learning
style penalty (trackmania_rl/style_constraint.py), so that a lap of the agent is measured exactly like a lap of a human driver.

A lap is described by LapSignals: where the car was (arc length and lateral offset along a centerline), how fast, and the three inputs
(brake, throttle, steering) on a 10 ms grid. corner_rows() turns that into one row of features per corner:
  b, f, l_entry, l_apex, l_exit, s            the style vector z of the problem formulation
  brake-zone features (bz_*), speeds, coasting, steering reversals, and a few extras

Only numpy and scipy are needed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

PERIOD_MS = 10  # resolution of the input timelines


# ─────────────────────────────────────────────────────────────────────────────
# Corners from the centerline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Corner:
    idx: int
    s_start: float
    s_end: float
    s_apex: float
    direction: int  # +1: centre of curvature on the +normal side, -1: other side
    angle_deg: float
    min_radius_m: float


@dataclass
class Centerline:
    pts: np.ndarray  # (n, 3)
    s: np.ndarray  # arc length at every point
    tangent: np.ndarray  # (n, 2) unit tangent in the x-z plane
    normal: np.ndarray  # (n, 2) tangent rotated +90 degrees
    heading: np.ndarray  # smoothed unwrapped heading
    curvature: np.ndarray  # signed 1/m, > 0 turns towards +normal


def build_centerline(pts: np.ndarray, smooth_m: float = 6.0) -> Centerline:
    xz = pts[:, [0, 2]]
    seg = np.diff(xz, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    ds = float(seg_len.mean())
    heading = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    heading = np.concatenate([heading, heading[-1:]])
    heading_s = gaussian_filter1d(heading, smooth_m / ds, mode="nearest")
    curvature = np.gradient(heading_s, s)
    tangent = np.stack([np.cos(heading_s), np.sin(heading_s)], axis=1)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return Centerline(pts, s, tangent, normal, heading_s, curvature)


def detect_corners(
    cl: Centerline,
    k_thresh: float = 1 / 150,
    merge_gap_m: float = 25.0,
    min_len_m: float = 20.0,
    min_angle_deg: float = 20.0,
) -> list:
    """Contiguous same-sign runs of |curvature| > k_thresh, merged across short gaps."""
    kappa = cl.curvature
    sign = np.where(kappa > k_thresh, 1, np.where(kappa < -k_thresh, -1, 0))
    runs = []  # [start_idx, end_idx, sign]
    i, n = 0, len(sign)
    while i < n:
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sign[j + 1] == sign[i]:
            j += 1
        runs.append([i, j, int(sign[i])])
        i = j + 1

    merged = []
    for r in runs:
        if merged and merged[-1][2] == r[2] and cl.s[r[0]] - cl.s[merged[-1][1]] < merge_gap_m:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    corners = []
    for a, b, sg in merged:
        length = cl.s[b] - cl.s[a]
        angle = abs(np.degrees(cl.heading[b] - cl.heading[a]))
        if length < min_len_m or angle < min_angle_deg:
            continue
        apex = a + int(np.argmax(np.abs(kappa[a : b + 1])))
        corners.append(
            Corner(
                idx=len(corners) + 1,
                s_start=float(cl.s[a]),
                s_end=float(cl.s[b]),
                s_apex=float(cl.s[apex]),
                direction=sg,
                angle_deg=float(angle),
                min_radius_m=float(1.0 / max(abs(kappa[apex]), 1e-6)),
            )
        )
    return corners


# ─────────────────────────────────────────────────────────────────────────────
# Position on the track and input features
# ─────────────────────────────────────────────────────────────────────────────


def project_on_centerline(positions: np.ndarray, speeds_ms: np.ndarray, cl: Centerline, dt: float = 0.1):
    """
    Sequential projection: each sample may only advance a bounded distance past the previous
    one, so sections of the track that pass close to each other cannot be confused.
    Returns arc length p and signed lateral offset e (+ = along cl.normal).
    """
    n = len(positions)
    ds = float(cl.s[-1] / (len(cl.s) - 1))
    xz = positions[:, [0, 2]]
    cxz = cl.pts[:, [0, 2]]
    p = np.zeros(n)
    e = np.zeros(n)
    prev = 0
    for i in range(n):
        reach = int((speeds_ms[i] * dt * 3.0 + 25.0) / ds)
        lo, hi = max(0, prev - int(10 / ds)), min(len(cxz), prev + reach + 1)
        d = np.linalg.norm(cxz[lo:hi] - xz[i], axis=1)
        k = lo + int(np.argmin(d))
        rel = xz[i] - cxz[k]
        p[i] = cl.s[k] + float(rel @ cl.tangent[k])
        e[i] = float(rel @ cl.normal[k])
        prev = k
    return p, e



def _reversals_u(u: np.ndarray, deadband: float = 0.1) -> int:
    d = np.sign(u[np.abs(u) > deadband])
    return int(np.count_nonzero(np.diff(d) != 0)) if len(d) > 1 else 0




BRAKE_WINDOW_M = 200.0  # how far before a corner entry a brake press still belongs to that corner
BRAKE_CHAIN_GAP = 40  # samples (0.4 s): presses closer than this are one braking zone
BRAKE_MIN_SCRUB_KMH = 8.0  # a zone must take off at least this much speed and be at least BRAKE_MIN_LEN_M long
BRAKE_MIN_LEN_M = 10.0


def _brake_zone(brake, p10, v10, c, prev_end, idx_at, peak_v) -> dict:
    """
    The braking zone of one corner: among the chains of brake presses between max(entry - BRAKE_WINDOW_M,
    end of the previous corner) and the apex, the one that takes off the most speed.
      bz_onset_m    metres before the corner entry where braking starts
      bz_release_m  metres after the corner entry where braking ends (> 0 means braking into the corner, trail braking)
      bz_len_m      length of the zone
      bz_presses    separate presses inside the zone (1 = one clean press, more = pumping)
      bz_scrub_kmh  speed lost between onset and release
      bz_onset_speed speed at onset / peak speed of the lap
    All NaN when the corner is not braked (no chain takes off 8 km/h over 10 m); bz_taps_other counts the other chains.
    """
    out = dict(braked=0, bz_onset_m=np.nan, bz_release_m=np.nan, bz_len_m=np.nan, bz_presses=np.nan, bz_scrub_kmh=np.nan, bz_onset_speed=np.nan, bz_taps_other=0.0)
    lo, hi = idx_at(max(c.s_start - BRAKE_WINDOW_M, prev_end)), idx_at(c.s_apex)
    seg = brake[lo : hi + 1].astype(np.int8)
    if not seg.any():
        return out
    d = np.diff(np.concatenate([[0], seg, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]  # run i covers samples starts[i]..ends[i]-1
    chains, first = [], 0
    for r in range(1, len(starts) + 1):
        if r == len(starts) or starts[r] - ends[r - 1] >= BRAKE_CHAIN_GAP:
            chains.append((first, r - 1))
            first = r
    best = None
    for f, l in chains:
        i_on, i_off = lo + starts[f], lo + ends[l] - 1
        scrub = float(v10[i_on] - v10[i_off])
        if best is None or scrub > best[0]:
            best = (scrub, f, l, i_on, i_off)
    scrub, f, l, i_on, i_off = best
    length = float(p10[i_off] - p10[i_on])
    out["bz_taps_other"] = float(sum(ends[ll] - starts[ff] > 0 for ff, ll in chains) - 1)  # other chains in the window
    if scrub < BRAKE_MIN_SCRUB_KMH or length < BRAKE_MIN_LEN_M:
        return out  # a tap, not a braking zone
    out.update(
        braked=1,
        bz_onset_m=float(c.s_start - p10[i_on]),
        bz_release_m=float(p10[i_off] - c.s_start),
        bz_len_m=length,
        bz_presses=float(l - f + 1),
        bz_scrub_kmh=scrub,
        bz_onset_speed=float(v10[i_on]) / peak_v,
    )
    return out




# ─────────────────────────────────────────────────────────────────────────────
# One lap -> per-corner features
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LapSignals:
    """
    One lap on the track.

    t_s, p, e, speed_kmh are samples (any spacing, 100 ms for replays, one per decision for the agent):
    time in seconds, arc length along the centerline (m), signed lateral offset (m, + = along the centerline normal) and speed (km/h).
    brake, accel, u are the inputs on a PERIOD_MS grid starting at race time 0: brake and accel are bool, u is the steering in [-1, 1], + = right.
    """

    t_s: np.ndarray
    p: np.ndarray
    e: np.ndarray
    speed_kmh: np.ndarray
    brake: np.ndarray
    accel: np.ndarray
    u: np.ndarray
    race_time_ms: int

    def __post_init__(self):
        self.n10 = int(self.race_time_ms // PERIOD_MS + 1)
        self.t10 = np.arange(self.n10) * PERIOD_MS / 1000.0
        self.p_strict = np.maximum.accumulate(self.p)
        self.p_strict = self.p_strict + np.arange(len(self.p_strict)) * 1e-6  # strictly increasing, for the inverse interpolation
        self.p10 = np.interp(self.t10, self.t_s, self.p_strict)
        self.e10 = np.interp(self.t10, self.t_s, self.e)
        self.v10 = np.interp(self.t10, self.t_s, self.speed_kmh)
        self.peak_v = max(float(self.v10.max()), 1.0)

    def t_at(self, s_pos: float) -> float:
        return float(np.interp(s_pos, self.p_strict, self.t_s))

    def idx_at(self, s_pos: float) -> int:
        return int(np.clip(round(self.t_at(s_pos) * 1000 / PERIOD_MS), 0, self.n10 - 1))

    def e_at(self, s_pos: float) -> float:
        return float(np.interp(s_pos, self.p10, self.e10))


def corner_rows(sig: LapSignals, corners: list, half_width: float, approach_m: float, meta: dict) -> list:
    """One dict per corner with the style features of this lap; `meta` (lap id, driver, lap time, ...) is copied into every row."""
    brake, accel, u, v10, e10 = sig.brake, sig.accel, sig.u, sig.v10, sig.e10
    idx_at, e_at = sig.idx_at, sig.e_at
    rows = []
    for ci, c in enumerate(corners):
        ia, ib, iap = idx_at(c.s_start), idx_at(c.s_end), idx_at(c.s_apex)
        row = dict(meta, corner=c.idx)
        if ib - ia < 3:
            rows.append(row)
            continue
        dur = (ib - ia) * PERIOD_MS / 1000.0
        sl = slice(ia, ib)
        sgn = c.direction / half_width
        row.update(
            b=float(brake[sl].mean()),
            f=float((accel[sl] & ~brake[sl]).mean()),
            l_entry=e_at(c.s_start) * sgn,
            l_apex=e_at(c.s_apex) * sgn,
            l_exit=e_at(c.s_end) * sgn,
            l_mean=float(e10[sl].mean()) * sgn,
            s=float(np.abs(np.diff(u[ia : ib + 1])).sum()) / dur,
            steer_frac=float((np.abs(u[sl]) > 0.05).mean()),
            steer_reversals=_reversals_u(u[sl]),
            coast=float((~accel[sl] & ~brake[sl]).mean()),
            v_entry=float(v10[ia]),
            v_apex=float(v10[iap]),
            v_min=float(v10[sl].min()),
            v_exit=float(v10[ib]),
            v_entry_ratio=float(v10[ia]) / sig.peak_v,
            corner_time_s=dur,
        )
        # braking before the corner: how early and how much
        pa = idx_at(max(c.s_start - approach_m, 0.0))
        app = brake[pa:ia]
        row["b_approach"] = float(app.mean()) if len(app) else np.nan
        if app.any():
            first = pa + int(np.argmax(app))
            row["brake_onset_m"] = float(c.s_start - np.interp(first * PERIOD_MS / 1000.0, sig.t_s, sig.p_strict))
        else:
            row["brake_onset_m"] = 0.0
        row.update(_brake_zone(brake, sig.p10, v10, c, corners[ci - 1].s_end if ci else 0.0, idx_at, sig.peak_v))
        rows.append(row)
    return rows
