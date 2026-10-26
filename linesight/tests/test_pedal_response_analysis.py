"""
Offline checks of the analysis part of scripts/tools/pedal_response_test.py (the part that talks to the game is run by hand).
Run from linesight/:  python tests/test_pedal_response_analysis.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tools"))
from pedal_response_test import acceleration_kmh_per_s, duty_mask, response_curve, scripted_action  # noqa: E402
from trackmania_rl.action_adapter import ActionSpec, game_command  # noqa: E402


def test_acceleration_is_the_slope_of_speed():
    t = np.arange(0, 1.2, 0.04)
    assert np.isclose(acceleration_kmh_per_s(t, 80 + 35.0 * t), 35.0)
    assert np.isclose(acceleration_kmh_per_s(t, 150 - 90.0 * t + np.random.default_rng(0).normal(0, 0.01, len(t))), -90.0, atol=0.5)


def test_response_curve():
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    linear = response_curve(levels, [0.0, -20.0, -40.0, -60.0, -80.0])  # a brake that decelerates in proportion to the level
    assert all(np.isclose(linear[lv], lv) for lv in levels)
    saturating = response_curve(levels, [0.0, -60.0, -75.0, -78.0, -80.0])  # a pedal that does almost everything with a quarter of the ticks
    assert saturating[0.25] > 0.7 and saturating[1.0] == 1.0 and saturating[0.0] == 0.0
    for bad in ([0.25, 0.5], ):
        try:
            response_curve(bad, [1, 2])
            raise AssertionError("levels without 0 and 1 must be refused")
        except ValueError:
            pass


def test_tick_patterns():
    assert [format(duty_mask(0.5, 4, p), "04b")[::-1] for p in ("spread", "front", "back")] == ["1010", "1100", "0011"]  # bit i = tick i
    assert [duty_mask(1.0, 4, p) for p in ("spread", "front", "back")] == [0b1111] * 3
    assert [duty_mask(0.0, 4, p) for p in ("spread", "front", "back")] == [0] * 3
    # the "spread" pattern is exactly what the training code sends to the game
    spec = ActionSpec(4)
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert duty_mask(level, 4, "spread") == game_command(np.array([0.0, level, level]), spec)[1] == game_command(np.array([0.0, level, level]), spec)[2]


def test_scripted_action_has_a_run_up_then_the_test_level():
    assert list(scripted_action(0, "brake", 0.75, switch_at=10, steer=0.1)) == [np.float32(0.1), 1.0, -1.0]
    assert list(scripted_action(10, "brake", 0.75, 10, 0.0)) == [0.0, -1.0, 0.5]  # brake level 0.75 -> raw 0.5, throttle released
    assert list(scripted_action(10, "throttle", 0.25, 10, 0.0)) == [0.0, -0.5, -1.0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
