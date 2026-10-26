"""
Offline checks for the analog steering levels (no game needed). Run from linesight/:  python tests/test_analog_inputs.py
They cannot check that the game accepts InputType::Steer; that has to be tried once in TMInterface.
"""

import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_files.inputs_list import make_inputs  # noqa: E402

LEGACY = [  # the original 12 actions: (left, right, accelerate, brake)
    (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0),
    (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0),
    (0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1),
    (0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1),
]  # fmt: skip


def test_default_is_the_original_list():
    inputs, forward, backward, n_forward = make_inputs()
    assert [(a["left"], a["right"], a["accelerate"], a["brake"]) for a in inputs] == [tuple(map(bool, t)) for t in LEGACY]
    assert (forward, backward, n_forward) == (0, 6, 3)


def test_five_levels():
    inputs, forward, backward, n_forward = make_inputs([-1, -0.5, 0, 0.5, 1])
    assert len(inputs) == 20 and (forward, backward, n_forward) == (0, 10, 5)
    assert [a["steer"] for a in inputs[:5]] == [0.0, -1.0, 1.0, -0.5, 0.5]
    half_left = next(a for a in inputs if a["steer"] == -0.5)
    assert half_left["left"] and not half_left["right"]


def test_levels_are_validated():
    for bad in ([-1, 1], [-2, 0, 2]):
        try:
            make_inputs(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_message_ids_agree_between_plugin_and_client():
    as_src = (ROOT / "trackmania_rl/tmi_interaction/Python_Link.as").read_text()
    as_ids = {k: int(v) for k, v in re.findall(r"^\s*(\w+)\s*=\s*(\d+),", as_src.split("enum MessageType")[1].split("}")[0], re.M)}
    py_src = (ROOT / "trackmania_rl/tmi_interaction/tminterface2.py").read_text()
    py_names = re.findall(r"^\s+([A-Z_]+) = auto\(\)", py_src.split("class MessageType")[1].split("class TMInterface")[0], re.M)
    py_ids = {name: i + 1 for i, name in enumerate(py_names)}
    for k, v in as_ids.items():
        assert py_ids[re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", k).upper()] == v, k
    assert len(struct.pack("=iBBi", as_ids["CSetInputStateAnalog"], 1, 0, -32768)) == 4 + 1 + 1 + 4  # what the plugin reads


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
