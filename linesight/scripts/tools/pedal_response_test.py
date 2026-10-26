"""
Measures how the car answers a pedal that is played as a duty cycle, in the game. Run it once before training SAC.

TMNF has no analog pedals. The SAC agent chooses a throttle and a brake level in [0, 1], and trackmania_rl.action_adapter plays level k / 4 as
"pedal down on k of the 4 physics ticks (10 ms each) of a 40 ms decision". The agent can only use these levels if the car responds to
them roughly like a partial pedal: acceleration (or deceleration) that grows steadily with the level. This script checks that.

For every pedal (throttle, brake), pedal level and tick pattern it starts the map, drives full throttle in a straight line for --accelerate-s seconds,
then plays the level for --test-s seconds and fits the speed slope. It writes a table (CSV) and prints, per pedal and pattern,

    a(level)        mean acceleration in km/h per second (negative for the brake)
    response        (a(level) - a(0)) / (a(1) - a(0))     0 for the pedal up, 1 for the pedal fully down; a partial pedal gives about `level`

Tick patterns (what the plugin plays within the 4 ticks of a decision), for level 0.5:
    spread  [1, 0, 1, 0]   what action_adapter uses
    front   [1, 1, 0, 0]
    back    [0, 0, 1, 1]

Needs the game, TMInterface and the updated Python_Link.as (copied to TMInterface by this script). The car must be able to go straight for about
accelerate_s + test_s seconds from the start: use a map that starts with a long straight (the first map of map_cycle is used; --steer can hold
a small constant steering to keep a cambered start straight).

    python scripts/tools/pedal_response_test.py --out ../pedal_response.csv
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PATTERNS = ("spread", "front", "back")


# ─── analysis (no game needed; tests/test_pedal_response_analysis.py) ──────────────────────────────────────────────────────────────────


def acceleration_kmh_per_s(time_s: np.ndarray, speed_kmh: np.ndarray) -> float:
    """Slope of the straight line fitted to speed over time."""
    return float(np.polyfit(np.asarray(time_s, dtype=float), np.asarray(speed_kmh, dtype=float), 1)[0])


def response_curve(levels: Sequence[float], accelerations: Sequence[float]) -> Dict[float, float]:
    """(a(level) - a(0)) / (a(1) - a(0)) for every level; a pedal that acts like a partial pedal gives about the level itself."""
    levels = [float(x) for x in levels]
    acc = dict(zip(levels, (float(a) for a in accelerations)))
    if 0.0 not in acc or 1.0 not in acc:
        raise ValueError("the levels must include 0 and 1")
    span = acc[1.0] - acc[0.0]
    if abs(span) < 1e-9:
        raise ValueError("the pedal at 1 has the same effect as at 0: nothing to compare")
    return {lv: (a - acc[0.0]) / span for lv, a in acc.items()}


def duty_mask(level: float, ticks: int, pattern: str) -> int:
    """Bit i set = pedal down on tick i, for the three tick patterns above."""
    k = int(round(level * ticks))
    if pattern == "front":
        return (1 << k) - 1
    if pattern == "back":
        return ((1 << k) - 1) << (ticks - k)
    if pattern == "spread":
        from trackmania_rl.action_adapter import duty_pattern, pattern_to_mask

        return pattern_to_mask(duty_pattern(level, ticks))
    raise ValueError(pattern)


def scripted_action(k: int, pedal: str, level: float, switch_at: int, steer: float) -> np.ndarray:
    """Network-space action [steer, throttle, brake] in [-1, 1] for decision k: full throttle until switch_at, then the test level of `pedal`."""
    if k < switch_at:
        return np.array([steer, 1.0, -1.0], dtype=np.float32)
    if pedal == "throttle":
        return np.array([steer, 2 * level - 1, -1.0], dtype=np.float32)
    return np.array([steer, -1.0, 2 * level - 1], dtype=np.float32)


# ─── in the game ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def run_in_game(args) -> List[dict]:
    from torch.multiprocessing import Lock

    from config_files import config_copy
    from trackmania_rl.action_adapter import ActionSpec, from_policy_output, quantize
    from trackmania_rl.map_loader import load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager

    base_dir = Path(__file__).resolve().parents[2]
    config_copy.target_python_link_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_dir / "trackmania_rl" / "tmi_interaction" / "Python_Link.as", config_copy.target_python_link_path)

    spec = ActionSpec(ticks_per_action=4)
    dt = spec.decision_ms / 1000.0
    map_name, map_path, zone_centers_filename, _, _ = next(iter(config_copy.map_cycle[0]))
    zone_centers = load_next_map_zone_centers(zone_centers_filename, base_dir)
    manager = game_instance_manager.GameInstanceManager(
        game_spawning_lock=Lock(),
        running_speed=config_copy.running_speed,
        run_steps_per_action=spec.ticks_per_action,
        max_overall_duration_ms=int((args.accelerate_s + args.test_s + 0.4) * 1000),
        max_minirace_duration_ms=int((args.accelerate_s + args.test_s + 5) * 1000),
        tmi_port=config_copy.base_tmi_port,
    )
    switch_at = int(round(args.accelerate_s / dt))
    end_at = switch_at + int(round(args.test_s / dt))
    rows = []
    for pedal in ("throttle", "brake"):
        for pattern in args.patterns:
            for level in args.levels:
                for repeat in range(args.repeats):
                    state = {"k": 0, "pattern": pattern, "pedal": pedal, "level": level}

                    def play(raw_action, spec_):  # the manager's request_continuous_inputs, with the tick pattern under test
                        action = quantize(from_policy_output(raw_action), spec_)
                        masks = {p: duty_mask(action[1 if p == "throttle" else 2], spec_.ticks_per_action, state["pattern"] if state["k"] >= switch_at else "spread") for p in ("throttle", "brake")}
                        manager.iface.set_input_pattern(int(round(action[0] * 65536)), spec_.ticks_per_action, masks["throttle"], masks["brake"])
                        return action

                    def policy(frame, floats):
                        raw = scripted_action(state["k"], state["pedal"], state["level"], switch_at, args.steer)
                        state["k"] += 1
                        return raw, True, 0.0, np.zeros(1, dtype=np.float32)

                    manager.request_continuous_inputs = play
                    results, _ = manager.rollout(policy, map_path, zone_centers, lambda: None, continuous_spec=spec)
                    speeds = np.asarray(results["speeds_kmh"], dtype=float)
                    if len(speeds) <= switch_at + 3:
                        print(f"  {pedal} {pattern} level {level}: the run ended after {len(speeds)} decisions, before the test phase; skipped")
                        continue
                    window = slice(switch_at + 2, min(end_at, len(speeds)))  # skip 80 ms for the first pattern to take effect
                    t = np.arange(len(speeds))[window] * dt
                    a = acceleration_kmh_per_s(t, speeds[window])
                    rows.append(dict(pedal=pedal, pattern=pattern, level=level, repeat=repeat, accel_kmh_per_s=a, speed_at_switch_kmh=float(speeds[switch_at]), n_samples=int(window.stop - window.start)))
                    print(f"  {pedal:8s} {pattern:7s} level {level:4.2f} (run {repeat + 1}): {a:+7.1f} km/h/s, speed at switch {speeds[switch_at]:.0f} km/h")
    return rows


def summarize(rows: List[dict]) -> None:
    for pedal in ("throttle", "brake"):
        for pattern in PATTERNS:
            sub = [r for r in rows if r["pedal"] == pedal and r["pattern"] == pattern]
            levels = sorted({r["level"] for r in sub})
            if not {0.0, 1.0} <= set(levels):
                continue
            acc = [float(np.mean([r["accel_kmh_per_s"] for r in sub if r["level"] == lv])) for lv in levels]
            curve = response_curve(levels, acc)
            print(f"\n{pedal} / {pattern}")
            print("  level   a (km/h/s)   response   (partial-pedal behaviour: response close to level)")
            for lv, a in zip(levels, acc):
                print(f"  {lv:5.2f}   {a:+9.1f}   {curve[lv]:8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("pedal_response.csv"))
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--patterns", nargs="+", choices=PATTERNS, default=list(PATTERNS))
    ap.add_argument("--accelerate-s", type=float, default=3.0, help="full throttle before the test phase")
    ap.add_argument("--test-s", type=float, default=1.2, help="length of the test phase")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--steer", type=float, default=0.0, help="constant steering in [-1, 1] to keep the car straight")
    args = ap.parse_args()
    rows = run_in_game(args)
    if not rows:
        sys.exit("no measurements")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    summarize(rows)


if __name__ == "__main__":
    # train.py's convention: config.py is copied to config_copy.py before anything imports it
    base_dir = Path(__file__).resolve().parents[2]
    shutil.copyfile(base_dir / "config_files" / "config.py", base_dir / "config_files" / "config_copy.py")
    main()
