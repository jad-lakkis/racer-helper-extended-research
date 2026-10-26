#!/usr/bin/env python3
"""
Extract three driver-style parameters from a Trackmania .Replay.Gbx file.

Outputs (printed + returned as dict):
  braking_aggression       float [0, 1]
  oversteer_understeer     float [-5, 5]
  corner_entry_speed_level str   low | medium | high

Requirements:  pip install pygbx scipy
Usage:
  cd linesight/
  PYTHONPATH=. python scripts/extract_driver_profile.py <replay.Replay.Gbx>
"""

import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────
SLIP_SAT              = 0.3     # lateral/forward ratio where oversteer signal → 1
# Matches reward-function constant (_UNDERSTEER_SLIP_MAX in buffer_management.py).
# Fires on ~42 % of steering samples in the temporal-smoothing frame (smooth cornering
# gives v_lat/v_fwd ≈ 0–0.04). The resulting negative bias is intentional: a clean
# non-sliding driver should read as mildly understeer.
UNDERSTEER_MAX        = 0.040
SPEED_MIN_MPS         = 10.0    # ignore samples below ~36 km/h
CORNER_CURV_THRESH    = 0.010   # 1/m — min path curvature for a corner (≈100 m radius)
                                # position-based curvature; spline-based used 0.025 but
                                # was dominated by k=1 kink artifacts (see _pos_curvature)
CORNER_MIN_GAP        = 5       # min samples between corner-entry detections
BRAKING_LOOKAHEAD_M   = 200.0   # metres ahead to scan for corners (braking metric)
BRAKING_SPEED_MIN_KMH = 50      # only score braking decisions above this speed


# ─────────────────────────────────────────────────────────────────────────────
# GBX loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_ghost(gbx_path: str):
    try:
        from pygbx import Gbx, GbxType
    except ImportError:
        sys.exit("[ERROR] pygbx not installed.  Run: pip install pygbx")

    g = Gbx(str(gbx_path))
    ghosts = g.get_classes_by_ids([GbxType.CTN_GHOST])
    if not ghosts:
        sys.exit("[ERROR] No ghost data found in replay.")

    ghost = min(ghosts, key=lambda gh: gh.cp_times[-1])  # best ghost in file
    race_start_ms = next(
        (e.time for e in ghost.control_entries
         if e.event_name == "_FakeIsRaceRunning" and e.enabled),
        ghost.control_entries[0].time,
    )
    records_to_keep = round(ghost.race_time / ghost.sample_period)
    return ghost, race_start_ms, ghost.race_time, records_to_keep


# ─────────────────────────────────────────────────────────────────────────────
# Path curvature from positions (kink-free)
# ─────────────────────────────────────────────────────────────────────────────

def _pos_curvature(positions: np.ndarray, disp_speed: np.ndarray,
                   n_samples: int) -> np.ndarray:
    """
    Kink-free path curvature κ[i] = |dθ/dt| / speed  (1/m).

    Computed from the heading angle θ = atan2(v_z, v_x) of the velocity
    vector, itself derived from finite differences of the position record.
    A light Gaussian smooth (σ = 1.5 steps / 150 ms) on the unwrapped
    heading suppresses the noise inherent in 100 ms position sampling
    before the yaw-rate derivative is taken.

    Why not use the k=1 centerline tangent differences instead?
    The k=1 spline creates a direction kink at every original 100 ms sample
    (~every 7 m at race speed). These kinks produce spurious curvature spikes
    of 0.025–0.37 1/m throughout the path, inflating the corner count ~6×
    and making the corner-entry-speed and braking-context metrics unreliable.
    """
    dt = 0.1
    pos = positions[:n_samples]

    vel_xz = np.zeros((n_samples, 2))
    vel_xz[1:-1] = (pos[2:, [0, 2]] - pos[:-2, [0, 2]]) / (2 * dt)
    vel_xz[0]    = (pos[1,  [0, 2]] - pos[0,  [0, 2]]) / dt
    vel_xz[-1]   = (pos[-1, [0, 2]] - pos[-2, [0, 2]]) / dt

    heading   = np.arctan2(vel_xz[:, 1], vel_xz[:, 0])
    heading_s = gaussian_filter1d(np.unwrap(heading), sigma=1.5)

    yaw_rate = np.zeros(n_samples)
    yaw_rate[1:-1] = (heading_s[2:] - heading_s[:-2]) / (2 * dt)
    yaw_rate[0]    = yaw_rate[1]
    yaw_rate[-1]   = yaw_rate[-2]

    speed_ms = np.maximum(disp_speed[:n_samples] / 3.6, 1.0)
    return np.abs(yaw_rate) / speed_ms


# ─────────────────────────────────────────────────────────────────────────────
# Steer state timeline
# ─────────────────────────────────────────────────────────────────────────────

def decode_analog_steer(entry) -> float:
    """
    'Steer' events store a signed 24-bit value split over the two uint16 fields
    (flags << 16 | enabled); +-65536 is full lock. Keyboard drivers on newer replays
    also use this event with the values -65536 / 0 / +65536.
    """
    raw = ((entry.flags << 16) | entry.enabled) & 0xFFFFFF
    return ((raw ^ 0x800000) - 0x800000) / 65536.0


def _steer_timeline(control_entries, race_start_ms: int,
                    n: int, period_ms: int) -> np.ndarray:
    result = np.zeros(n, dtype=bool)
    analog = sorted([(e.time - race_start_ms, abs(decode_analog_steer(e)) > 0.05)
                     for e in control_entries if e.event_name == "Steer"],
                    key=lambda x: x[0])
    if analog:
        active, ptr = False, 0
        for i in range(n):
            while ptr < len(analog) and analog[ptr][0] <= i * period_ms:
                active = analog[ptr][1]; ptr += 1
            result[i] = active
        return result
    for name in ("SteerLeft", "SteerRight"):
        evts = sorted(
            [(e.time - race_start_ms, bool(e.enabled))
             for e in control_entries if e.event_name == name],
            key=lambda x: x[0])
        active, ptr = False, 0
        for i in range(n):
            while ptr < len(evts) and evts[ptr][0] <= i * period_ms:
                active = evts[ptr][1]; ptr += 1
            result[i] |= active
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Braking aggression (context-aware)
# ─────────────────────────────────────────────────────────────────────────────

def compute_braking_aggression(control_entries, race_start_ms: int,
                                race_time_ms: int, n_race_samples: int,
                                disp_speed: np.ndarray,
                                pos_curv: np.ndarray) -> float:
    """
    Braking frequency conditioned on corner-approach context:
      numerator   = samples where brake is pressed AND context is active
      denominator = samples where context is active

    Context = car is above BRAKING_SPEED_MIN_KMH AND a corner exists within
    the next BRAKING_LOOKAHEAD_M metres ahead.

    Distance is converted to a time-based lookahead at the current speed so
    the window covers the same physical distance regardless of velocity.

    Falls back to speed²-weighted braking fraction when no context samples
    exist (e.g. pure-speed tracks with no detectable corners).
    """
    period_ms = 100
    dt = period_ms / 1000.0

    brake_events = sorted(
        [(e.time - race_start_ms, bool(e.enabled))
         for e in control_entries if e.event_name == "Brake"],
        key=lambda x: x[0])
    brake = np.zeros(n_race_samples, dtype=bool)
    active, ptr = False, 0
    for i in range(n_race_samples):
        while ptr < len(brake_events) and brake_events[ptr][0] <= i * period_ms:
            active = brake_events[ptr][1]; ptr += 1
        brake[i] = active

    context = np.zeros(n_race_samples, dtype=bool)
    for i in range(n_race_samples):
        if disp_speed[i] < BRAKING_SPEED_MIN_KMH:
            continue
        speed_ms_i = max(disp_speed[i] / 3.6, 1.0)
        # convert 200m lookahead to sample count at current speed
        lookahead = int(BRAKING_LOOKAHEAD_M / speed_ms_i / dt) + 1
        hi = min(i + lookahead, n_race_samples)
        if hi > i and pos_curv[i:hi].max() > CORNER_CURV_THRESH:
            context[i] = True

    denom = context.sum()
    if denom > 0:
        return float(brake[context].sum() / denom)

    # Fallback: speed²-weighted brake fraction
    w = (disp_speed[:n_race_samples] / 3.6) ** 2
    total_w = w.sum()
    if total_w < 1e-9:
        return 0.0
    return float((w * brake).sum() / total_w)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — Oversteer / understeer score
# ─────────────────────────────────────────────────────────────────────────────

def compute_oversteer_understeer(positions: np.ndarray, disp_speed: np.ndarray,
                                  control_entries, race_start_ms: int,
                                  n_race_samples: int) -> float:
    """
    Gaussian-smoothed velocity (σ = 6 steps / 600 ms) as forward reference.

    For a slide event lasting 1-3 samples, smooth_vel[i] is ~93 % determined
    by non-slide neighbours, so the perpendicular component of raw vel captures
    genuine lateral slip — without the circular-reference bias of using the
    driver's own path as the tangent frame.

    o_signal ∈ [-1, 1]:
      +1  full oversteer  (slip ratio saturated at SLIP_SAT)
      -1  full understeer (steering active, slip ratio < UNDERSTEER_MAX)
       0  neutral

    Score = mean(o_signal) × 5  →  [-5, 5]
    """
    period_ms = 100
    dt = period_ms / 1000.0

    pos = positions[:n_race_samples]
    spd = disp_speed[:n_race_samples]

    vel_xz = np.zeros((n_race_samples, 2))
    vel_xz[1:-1] = (pos[2:, [0, 2]] - pos[:-2, [0, 2]]) / (2 * dt)
    vel_xz[0]    = (pos[1,  [0, 2]] - pos[0,  [0, 2]]) / dt
    vel_xz[-1]   = (pos[-1, [0, 2]] - pos[-2, [0, 2]]) / dt

    smooth_xz = gaussian_filter1d(vel_xz, sigma=6.0, axis=0)

    is_steer = _steer_timeline(control_entries, race_start_ms,
                                n_race_samples, period_ms)

    o_signals = []
    for i in range(1, n_race_samples - 1):
        if spd[i] / 3.6 < SPEED_MIN_MPS:
            continue
        sv      = smooth_xz[i]
        sv_norm = np.linalg.norm(sv)
        if sv_norm < 1.0:
            continue
        t     = sv / sv_norm
        p     = np.array([-t[1], t[0]])
        v     = vel_xz[i]
        v_fwd = abs(float(np.dot(v, t)))
        v_lat = abs(float(np.dot(v, p)))
        denom = max(v_fwd, 1.0)
        slip  = min(v_lat / denom, SLIP_SAT) / SLIP_SAT
        under = is_steer[i] and (v_lat / denom < UNDERSTEER_MAX)
        o_signals.append(max(-1.0, min(1.0, slip - (1.0 if under else 0.0))))

    return 0.0 if not o_signals else float(np.mean(o_signals)) * 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — Corner entry speed level
# ─────────────────────────────────────────────────────────────────────────────

def compute_corner_entry_speed(disp_speed: np.ndarray, n_race_samples: int,
                                pos_curv: np.ndarray) -> tuple:
    """
    Ratio = median corner-entry speed / driver's own peak speed.
      high   > 0.60  (late braking, carries maximum speed into corners)
      medium   0.35–0.60
      low    < 0.35  (conservative, significant speed scrubbed before corners)

    Corner entries are detected from the kink-free position-based curvature.
    The first 10 samples (1 s) are skipped to exclude the race-start
    acceleration phase.

    Returns (level_str, ratio_float) where ratio_float is the raw scalar
    used by the reward function (corner_entry_speed_ratio in config.py).
    """
    curv = pos_curv[:n_race_samples]
    above  = (curv > CORNER_CURV_THRESH).astype(int)
    rising = np.where(np.diff(above) == 1)[0]

    entry_speeds, last = [], -CORNER_MIN_GAP - 1
    for edge in rising:
        if edge < 10:               # skip race-start acceleration
            continue
        if edge - last < CORNER_MIN_GAP:
            continue
        entry_speeds.append(float(disp_speed[max(0, edge - 1)]))
        last = edge

    if len(entry_speeds) < 3:
        return f"unknown (only {len(entry_speeds)} corners detected)", 0.5

    ratio = float(np.median(entry_speeds)) / max(float(disp_speed[:n_race_samples].max()), 1.0)
    level = "high" if ratio > 0.60 else ("medium" if ratio > 0.35 else "low")
    return level, ratio


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def extract_driver_profile(gbx_path: str) -> dict:
    print(f"Replay : {Path(gbx_path).name}")

    ghost, race_start_ms, race_time_ms, n_race_samples = _load_ghost(gbx_path)
    print(f"Driver : {ghost.login}   race time: {race_time_ms / 1000:.3f}s   "
          f"respawns: {ghost.num_respawns}")

    positions = np.array([[r.position.x, r.position.y, r.position.z]
                           for r in ghost.records[:n_race_samples]])
    disp_spd  = np.array([r.display_speed for r in ghost.records[:n_race_samples]],
                          dtype=np.float64)

    pos_curv = _pos_curvature(positions, disp_spd, n_race_samples)

    braking = compute_braking_aggression(
        ghost.control_entries, race_start_ms, race_time_ms,
        n_race_samples, disp_spd, pos_curv)

    ous = compute_oversteer_understeer(
        positions, disp_spd, ghost.control_entries, race_start_ms,
        n_race_samples)

    corner_level, corner_ratio = compute_corner_entry_speed(disp_spd, n_race_samples, pos_curv)

    print()
    print("═" * 54)
    print("  DRIVER PROFILE")
    print("═" * 54)
    print(f"  braking_aggression         = {braking:.3f}")
    print(f"  oversteer_understeer_score = {ous:+.2f}")
    print(f"  corner_entry_speed_level   = {corner_level}")
    print(f"  corner_entry_speed_ratio   = {corner_ratio:.3f}")
    print()
    print("  ── paste into config_files/config.py ──")
    print(f"  braking_aggression         = {round(braking, 2)}")
    print(f"  oversteer_understeer_score = {round(ous, 1)}")
    print(f"  corner_entry_speed_ratio   = {round(corner_ratio, 2)}")
    print("═" * 54)

    return {
        "braking_aggression":         braking,
        "oversteer_understeer_score": ous,
        "corner_entry_speed_level":   corner_level,
        "corner_entry_speed_ratio":   corner_ratio,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: PYTHONPATH=. python scripts/extract_driver_profile.py <replay.Replay.Gbx>")
        sys.exit(1)
    extract_driver_profile(sys.argv[1])
