"""
Offline checks for trackmania_rl/action_adapter.py and the pattern message. Run from linesight/:  python tests/test_action_adapter.py
"""

import re
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trackmania_rl.action_adapter import (  # noqa: E402
    ActionSpec,
    duty_pattern,
    from_policy_output,
    game_command,
    input_timelines,
    mask_to_pattern,
    pattern_to_mask,
    quantize,
    to_policy_output,
)


def test_duty_patterns_for_four_ticks():
    expected = {0.0: [0, 0, 0, 0], 0.25: [1, 0, 0, 0], 0.5: [1, 0, 1, 0], 0.75: [1, 1, 1, 0], 1.0: [1, 1, 1, 1]}
    for level, pattern in expected.items():
        assert duty_pattern(level, 4).astype(int).tolist() == pattern, level


def test_duty_pattern_has_the_right_share_for_any_tick_count():
    for ticks in range(1, 17):
        for k in range(ticks + 1):
            p = duty_pattern(k / ticks, ticks)
            assert p.sum() == k, (ticks, k)
            if k > 0:
                assert p[0], (ticks, k)  # the pedal acts on the first tick of the decision
            if 0 < k < ticks:  # evenly spread: gaps between presses differ by at most 1
                idx = np.flatnonzero(p)
                gaps = np.diff(np.append(idx, ticks + idx[0]))
                assert gaps.max() - gaps.min() <= 1, (ticks, k, p)


def test_quantize_pedals_and_steering():
    spec = ActionSpec(ticks_per_action=4, steer_step=0.2)
    q = quantize(np.array([0.33, 0.3, 0.66]), spec)
    assert np.allclose(q, [0.4, 0.25, 0.75]), q
    q = quantize(np.array([-3.0, 1.7, -0.2]), spec)
    assert np.allclose(q, [-1.0, 1.0, 0.0]), q
    assert np.allclose(quantize(np.array([0.123, 0.5, 0.5]), ActionSpec())[0], 0.123)  # no steering grid


def test_masks_round_trip_and_game_command():
    for mask in range(16):
        assert pattern_to_mask(mask_to_pattern(mask, 4)) == mask
    assert game_command(np.array([0.5, 1.0, 0.5]), ActionSpec()) == (32768, 0b1111, 0b0101)
    assert game_command(np.array([-1.0, 0.0, 0.0]), ActionSpec()) == (-65536, 0, 0)


def test_policy_output_mapping_round_trips():
    a = np.array([0.3, 0.8, 0.1])
    assert np.allclose(from_policy_output(to_policy_output(a)), a)
    assert np.allclose(from_policy_output(np.array([-1, -1, 1])), [-1, 0, 1])


def test_timelines_have_the_requested_average_pedal():
    spec = ActionSpec()
    rng = np.random.default_rng(0)
    actions = np.column_stack([rng.uniform(-1, 1, 200), rng.integers(0, 5, 200) / 4, rng.integers(0, 5, 200) / 4])
    brake, accel, steer = input_timelines(actions, spec)
    assert len(brake) == 800 and len(steer) == 800
    assert np.allclose(brake.reshape(200, 4).mean(1), actions[:, 2])
    assert np.allclose(accel.reshape(200, 4).mean(1), actions[:, 1])
    assert np.allclose(steer.reshape(200, 4)[:, 0], actions[:, 0])


def test_pattern_message_agrees_between_plugin_and_client():
    as_src = (ROOT / "trackmania_rl/tmi_interaction/Python_Link.as").read_text()
    as_ids = {k: int(v) for k, v in re.findall(r"^\s*(\w+)\s*=\s*(\d+),", as_src.split("enum MessageType")[1].split("}")[0], re.M)}
    py_src = (ROOT / "trackmania_rl/tmi_interaction/tminterface2.py").read_text()
    py_names = re.findall(r"^\s+([A-Z_]+) = auto\(\)", py_src.split("class MessageType")[1].split("class TMInterface")[0], re.M)
    assert py_names.index("C_SET_INPUT_STATE_PATTERN") + 1 == as_ids["CSetInputStatePattern"]
    handler = as_src.split("case MessageType::CSetInputStatePattern:")[1].split("break;")[0]
    assert handler.count("ReadInt32()") == 4  # steer, ticks, throttle mask, brake mask
    assert len(struct.pack("=iiiii", 25, 0, 4, 0, 0)) == 4 + 4 * 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
