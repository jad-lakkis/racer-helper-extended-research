"""
Runs GameInstanceManager.rollout against a fake TMInterface: a car that drives the Hockolicious centerline at a constant speed and answers
the same messages as the game (run step, requested frame, checkpoint count). No game, no GPU. Run from linesight/:

    python tests/test_rollout_continuous.py          (needs torch, pygbx, scipy; cv2 / numba / psutil / xdo are stubbed if missing)

It checks the continuous (SAC) mode end to end: what is sent to the game for every decision, the recorded positions / executed actions / end
position, the previous-action and reference-line parts of the state, scoring and transitions of the finished rollout. It also runs the
original discrete mode on the same fake game to check that refactoring the state construction changed nothing.
"""

import shutil
import sys
import types
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LINESIGHT = ROOT / "linesight"
sys.path.insert(0, str(LINESIGHT))

# ─── stubs for what the real game needs and this machine does not have ──────────────────────────────────────────────────────────────
try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGRA2GRAY = 0
    cv2.cvtColor = lambda frame, code: frame[..., 0].copy()
    sys.modules["cv2"] = cv2
try:
    import numba  # noqa: F401
except ImportError:
    numba = types.ModuleType("numba")
    numba.njit = lambda f=None, **kw: (f if f is not None else (lambda g: g))
    sys.modules["numba"] = numba
try:
    import psutil  # noqa: F401
except ImportError:
    psutil = types.ModuleType("psutil")
    psutil.Process = object
    sys.modules["psutil"] = psutil
try:
    import xdo  # noqa: F401
except ImportError:
    xdo = types.ModuleType("xdo")
    xdo.Xdo = object
    sys.modules["xdo"] = xdo

CONFIG_COPY = LINESIGHT / "config_files" / "config_copy.py"
created_config_copy = not CONFIG_COPY.exists()
if created_config_copy:  # train.py makes this file at the start of a run; it is not tracked
    shutil.copyfile(LINESIGHT / "config_files" / "config.py", CONFIG_COPY)

from config_files import config_copy  # noqa: E402
from trackmania_rl import contact_materials  # noqa: E402
from trackmania_rl.action_adapter import ActionSpec, from_policy_output, game_command, quantize, to_policy_output  # noqa: E402
from trackmania_rl.driver_context import DriverPackage, RolloutStyleContext  # noqa: E402
from trackmania_rl.map_loader import load_next_map_zone_centers  # noqa: E402
from trackmania_rl.sac_rollout import rollout_to_transitions, score_rollout  # noqa: E402
from trackmania_rl.style_constraint import RewardWeights, StyleConstraint  # noqa: E402
from trackmania_rl.tmi_interaction import game_instance_manager  # noqa: E402
from trackmania_rl.tmi_interaction.tminterface2 import MessageType  # noqa: E402

PACKAGE = ROOT / "style_vectors" / "Hockolicious" / "packages" / "-wally-.npz"
MAP = "ESL-Hockolicious.Challenge.Gbx"
SPEED_MS = 55.0
MATERIAL_ID = next(iter(contact_materials.physics_behavior_fromint))


# ─── a fake game ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeGame:
    """A car that follows the centerline of the package at SPEED_MS with a sideways wobble; time advances by one decision per message."""

    def __init__(self, centerline, step_ms):
        self.cl = centerline
        self.step_ms = step_ms
        self.time_ms = -step_ms  # the first run step is at 0
        self.length = float(centerline.s[-1])
        self.finish_time_ms = int(1000 * self.length / SPEED_MS)
        self.frame_requested = False
        self.finish_sent = False
        self.finish_reached = False

    def car_state(self, time_ms):
        p = min(SPEED_MS * time_ms / 1000.0, self.length)
        i = min(int(np.searchsorted(self.cl.s, p)), len(self.cl.s) - 1)
        xz = self.cl.pts[i][[0, 2]] + self.cl.normal[i] * 3.0 * np.sin(p / 60.0)
        t = self.cl.tangent[i]
        return (float(xz[0]), float(self.cl.pts[i][1]), float(xz[1])), (float(SPEED_MS * t[0]), 0.0, float(SPEED_MS * t[1]))

    def sim_state(self, time_ms):
        pos, vel = self.car_state(time_ms)
        wheel = _Obj(real_time_state=_Obj(is_sliding=False, has_ground_contact=True, damper_absorb=0.01, contact_material_id=MATERIAL_ID))
        cp_times = [_Obj(time=time_ms)] if time_ms >= self.finish_time_ms else []
        return _Obj(
            race_time=int(time_ms),
            dyna=_Obj(current_state=_Obj(position=pos, rotation=_Obj(to_numpy=lambda: np.eye(3)), linear_speed=vel, angular_speed=(0.0, 0.1, 0.0))),
            scene_mobil=_Obj(gearbox_state=0, is_freewheeling=False, engine=_Obj(gear=3, actual_rpm=7000.0)),
            simulation_wheels=[wheel] * 4,
            cp_data=_Obj(cp_times=cp_times),
        )

    def advance(self, pending):
        if self.frame_requested:
            self.frame_requested = False
            pending.append(int(MessageType.SC_REQUESTED_FRAME_SYNC))
        elif self.finish_reached and not self.finish_sent:
            self.finish_sent = True
            self.time_ms = self.finish_time_ms
            pending.extend([int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC), 1, 1])
        else:
            self.time_ms += self.step_ms
            if self.time_ms + self.step_ms > self.finish_time_ms and self.time_ms >= 0 and not self.finish_reached:
                self.finish_reached = True  # the finish comes during the next stretch of 40 ms: the message is sent after this decision
            pending.extend([int(MessageType.SC_RUN_STEP_SYNC), self.time_ms])


class FakeIface:
    registered = True

    def __init__(self, game):
        self.game = game
        self.pending = deque()
        self.patterns = []  # set_input_pattern calls
        self.keys = []  # set_input_state calls
        self.speeds = []

    def _read_int32(self):
        if not self.pending:
            self.game.advance(self.pending)
        return self.pending.popleft()

    def _respond_to_call(self, msgtype):
        pass

    def get_simulation_state(self):
        return self.game.sim_state(self.game.time_ms)

    def request_frame(self, w, h):
        self.game.frame_requested = True

    def get_frame(self, w, h):
        frame = np.zeros((h, w, 4), dtype=np.uint8)
        frame[..., 0] = (self.game.time_ms // 40) % 251
        return frame

    def set_input_pattern(self, steer_int, ticks, throttle_mask, brake_mask):
        self.patterns.append((steer_int, ticks, throttle_mask, brake_mask))

    def set_input_state(self, **kw):
        self.keys.append(kw)

    def set_speed(self, speed):
        self.speeds.append(speed)

    def __getattr__(self, name):  # give_up, rewind_*, set_timeout, execute_command, toggle_interface, unrequest_frame, ...
        return lambda *args, **kwargs: None


def make_manager(package, step_ticks):
    manager = game_instance_manager.GameInstanceManager(
        game_spawning_lock=None, running_speed=1, run_steps_per_action=step_ticks, max_overall_duration_ms=300_000, max_minirace_duration_ms=2_000
    )
    game = FakeGame(package.style.centerline, 10 * step_ticks)
    manager.iface = FakeIface(game)
    manager.ensure_game_launched = lambda: None
    manager.last_rollout_crashed = True  # (the rollout asserts that it resumes a game that is waiting)
    manager.latest_map_path_requested = MAP
    manager.start_states[MAP] = object()
    manager.timeout_has_been_set = True
    manager.UI_disabled = True
    manager.game_activated = True
    return manager, game


def prepare_zone_centers(manager):
    zone_centers = load_next_map_zone_centers("ESL-Hockolicious_0.5m_cl2.npy", LINESIGHT)
    manager.next_real_checkpoint_positions = np.zeros((len(zone_centers), 3))
    manager.max_allowable_distance_to_real_checkpoint = np.full(len(zone_centers), np.inf)
    return zone_centers


# ─── tests ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def scripted_raw_action(k):
    throttle_level = (1.0, 0.75, 0.5, 0.25, 0.0)[k % 5]
    brake_level = 0.75 if k % 37 == 0 else 0.0
    return np.array([0.4 * np.sin(k / 30.0), 2 * throttle_level - 1, 2 * brake_level - 1], dtype=np.float32)


def test_continuous_rollout_against_a_fake_game():
    package = DriverPackage.load(PACKAGE)
    spec = ActionSpec(ticks_per_action=4)
    manager, game = make_manager(package, spec.ticks_per_action)
    zone_centers = prepare_zone_centers(manager)
    style_context = RolloutStyleContext(package, spec.decision_ms / 1000.0)

    calls = {"n": 0}
    raw_log = []

    def policy(frame, floats):
        assert frame.shape == (1, config_copy.H_downsized, config_copy.W_downsized) and frame.dtype == np.uint8
        raw = scripted_raw_action(calls["n"])
        calls["n"] += 1
        raw_log.append(raw)
        return raw, True, 0.0, np.zeros(1, dtype=np.float32)

    results, stats = manager.rollout(policy, MAP, zone_centers, lambda: None, continuous_spec=spec, style_context=style_context)

    n = calls["n"]
    assert stats["race_finished"] and abs(stats["race_time"] - game.finish_time_ms) < 50, stats.get("race_time")
    assert n > 1500 and len(results["positions"]) == len(results["executed_actions"]) == len(results["speeds_kmh"]) == n
    assert len(results["state_float"]) == n and len(results["frames"]) == n + 1 and np.isnan(results["frames"][-1])  # the finish adds a nan entry
    assert len(results["actions"]) == n + 1 and np.isnan(results["actions"][-1])
    assert results["reference_lap_id"] in {r.lap_id for r in package.refs}
    end = np.asarray(results["end_position"])
    assert np.linalg.norm(end[[0, 2]] - package.style.centerline.pts[-1][[0, 2]]) < 10

    # what was sent to the game: one pattern per decision, equal to the adapter's command for the executed action
    assert len(manager.iface.patterns) == n and not manager.iface.keys
    for raw, sent, executed in zip(raw_log, manager.iface.patterns, results["executed_actions"]):
        want = quantize(from_policy_output(raw), spec)
        assert np.allclose(executed, want)
        assert sent == (game_command(want, spec)[0], 4, game_command(want, spec)[1], game_command(want, spec)[2])
    assert {p[2] for p in manager.iface.patterns} == {0b1111, 0b0111, 0b0101, 0b0001, 0b0000}  # all five throttle levels were played

    # the state: original floats + 13 reference features; previous actions are the executed ones
    floats = np.stack(results["state_float"])
    assert floats.shape == (n, config_copy.sac_float_input_dim) == (n, config_copy.float_input_dim + 13) and np.isfinite(floats).all()
    assert list(floats[0, 1:21]) == [1.0, 0.0, 0.0, 0.0] * 5  # before the first decision: straight, full throttle
    for k in (1, 7, 100):
        steer, thr, brk = results["executed_actions"][k - 1]
        assert np.allclose(floats[k, 17:21], [thr, brk, max(0.0, -steer), max(0.0, steer)], atol=1e-6)
    ref = floats[:, -13:]
    assert np.abs(ref[:, :6]).max() < 1.5 and np.abs(ref[:, -1]).max() < 1.0  # line in half-widths, small error: the car follows the centerline

    # score and transitions
    constraint = StyleConstraint(package.style, spec, RewardWeights())
    score = score_rollout(constraint, results, stats, style_context.reference)
    assert score is not None and score.lap_time_ms == stats["race_time"] and not score.failed
    assert score.progress_m > 0.98 * package.style.centerline.s[-1] and np.isfinite(score.base_reward).all()
    tr = rollout_to_transitions(results, score, 0.1, 0.5, 3, 0.995, 10.0)
    assert len(tr["reward_n"]) == n and tr["img"].shape == (n, 1, config_copy.H_downsized, config_copy.W_downsized)
    assert tr["floats"].shape == (n, config_copy.sac_float_input_dim) and np.all(tr["discount_n"][-3:] == 0.0)
    # driving along the line earns progress reward; the fake car does not drive like the reference driver, so the style cost (first corners) outweighs it
    plain = rollout_to_transitions(results, score, 0.0, 0.0, 3, 0.995, 10.0)
    assert plain["reward_n"][:100].mean() > 0 and tr["reward_n"][:100].mean() < plain["reward_n"][:100].mean()
    assert score.style_total > 20  # a car that ignores the style ranges is charged for it


def test_discrete_rollout_still_builds_the_original_state():
    package = DriverPackage.load(PACKAGE)
    manager, game = make_manager(package, config_copy.tm_engine_step_per_action)
    zone_centers = prepare_zone_centers(manager)
    sequence = [3, 0, 7, 11, 5]
    calls = {"n": 0}

    def policy(frame, floats):
        idx = sequence[calls["n"] % len(sequence)]
        calls["n"] += 1
        return idx, True, 0.0, np.zeros(len(config_copy.inputs), dtype=np.float32)

    results, stats = manager.rollout(policy, MAP, zone_centers, lambda: None)
    n = calls["n"]
    assert stats["race_finished"] and n > 1000 and len(results["state_float"]) == n and "positions" not in results
    assert len(manager.iface.keys) == n and not manager.iface.patterns
    floats = np.stack(results["state_float"])
    assert floats.shape == (n, config_copy.float_input_dim)
    inputs = config_copy.inputs
    assert list(floats[0, 1:21]) == [value for _ in range(5) for value in (inputs[config_copy.action_forward_idx]["accelerate"], inputs[config_copy.action_forward_idx]["brake"], 0, 0)]
    for k in (1, 8, 51):
        previous = inputs[results["actions"][k - 1]]
        assert floats[k, 17:21].tolist() == [previous["accelerate"], previous["brake"], max(0.0, -previous["steer"]), max(0.0, previous["steer"])]
        assert manager.iface.keys[k - 1]["accelerate"] == previous["accelerate"]


if __name__ == "__main__":
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_"):
                fn()
                print("ok", name)
    finally:
        if created_config_copy and CONFIG_COPY.exists():
            CONFIG_COPY.unlink()
