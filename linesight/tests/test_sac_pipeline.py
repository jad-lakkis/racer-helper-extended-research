"""
Offline checks of the SAC pipeline around the game: state features of continuous actions, scoring a rollout, n-step transitions with the
adjusted reward, the learner (replay, multipliers, target laps, updates, save/load), the exploration policy and the .inputs writer.
Run from linesight/:  python tests/test_sac_pipeline.py   (needs torch; the replay-based parts also need pygbx and the Hockolicious replays)

A rollout of the game is replaced by a human lap played through the same arrays a rollout records (see test_style_constraint.py).
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "linesight"))
sys.path.insert(0, str(ROOT / "linesight" / "tests"))
from test_style_constraint import STYLE_DIR, _human_lap_as_a_rollout  # noqa: E402

from trackmania_rl import run_to_video  # noqa: E402
from trackmania_rl.action_adapter import ActionSpec, quantize, to_policy_output  # noqa: E402
from trackmania_rl.agents.sac import SACInferer, SACNet  # noqa: E402
from trackmania_rl.driver_context import N_REFERENCE_FEATURES, DriverPackage  # noqa: E402
from trackmania_rl.sac_learner import SACLearner  # noqa: E402
from trackmania_rl.sac_rollout import (  # noqa: E402
    DEFAULT_ACTION,
    SACExplorationPolicy,
    n_decisions,
    previous_action_features,
    rollout_to_transitions,
    score_rollout,
)
from trackmania_rl.style_constraint import RewardWeights, StyleConstraint  # noqa: E402

HW = (64, 64)
FLOAT_DIM = 10
PACKAGE = STYLE_DIR / "packages" / "-wally-.npz"
SPEC = ActionSpec(ticks_per_action=4)


def test_previous_action_features():
    assert previous_action_features([], 2) == [1.0, 0.0, 0.0, 0.0] * 2  # nothing yet: straight, full throttle
    history = [np.array([-0.5, 1.0, 0.0]), np.array([0.25, 0.5, 0.75])]
    # newest last; per action: throttle, brake, left amount, right amount
    assert previous_action_features(history, 3) == [1.0, 0.0, 0.0, 0.0] + [1.0, 0.0, 0.5, 0.0] + [0.5, 0.75, 0.0, 0.25]
    assert len(previous_action_features(history * 5, 5)) == 20
    assert list(DEFAULT_ACTION) == [0.0, 1.0, 0.0]


def _fake_rollout(frames_shape=(1, *HW)):
    """The recorded arrays of a rollout that is a human lap (or None without pygbx / replays)."""
    lap = _human_lap_as_a_rollout()
    if lap is None:
        return None
    pos, speed, actions, race_time = lap
    n = len(actions)
    rng = np.random.default_rng(0)
    executed = np.array([quantize(a, SPEC) for a in actions])
    results = dict(
        frames=[rng.integers(0, 255, frames_shape, dtype=np.uint8) for _ in range(n)] + [np.nan],  # a rollout that finished ends with a nan entry
        state_float=[rng.normal(size=FLOAT_DIM).astype(np.float32) for _ in range(n)],
        actions=[to_policy_output(a).astype(np.float32) for a in executed] + [np.nan],
        executed_actions=list(executed),
        positions=list(pos[:n].astype(np.float32)),
        speeds_kmh=list(speed[:n]),
        end_position=pos[n].astype(np.float32),
        reference_lap_id="x",
    )
    return results, dict(race_finished=True, race_time=race_time)


def _constraint():
    pkg = DriverPackage.load(PACKAGE)
    return pkg, StyleConstraint(pkg.style, SPEC, RewardWeights())


def test_score_rollout_matches_scoring_the_arrays_directly():
    fake = _fake_rollout()
    if fake is None:
        print("   skipped: needs pygbx and replays/tmnf_exchange/Hockolicious")
        return
    results, stats = fake
    pkg, constraint = _constraint()
    score = score_rollout(constraint, results, stats, pkg.refs[0])
    assert n_decisions(results) == len(results["executed_actions"]) and not score.failed and score.lap_time_ms == 54040
    pos = np.vstack([np.array(results["positions"], dtype=np.float64), results["end_position"][None]])
    direct = constraint.score(pos, np.append(results["speeds_kmh"], results["speeds_kmh"][-1]), np.array(results["executed_actions"]), True, 54040, pkg.refs[0])
    assert np.allclose(score.base_reward, direct.base_reward) and np.allclose(score.c_style, direct.c_style)

    unfinished = score_rollout(constraint, results, dict(race_finished=False), pkg.refs[0])  # cut short: no finish, no end position used
    assert unfinished.failed and unfinished.lap_time_ms is None and unfinished.c_fail[-1] == 1.0
    assert score_rollout(constraint, dict(positions=[0, 1], executed_actions=[0, 1], speeds_kmh=[0, 0]), stats, None) is None  # too short to say anything


def test_transitions_use_the_adjusted_reward_and_end_without_bootstrap():
    fake = _fake_rollout()
    if fake is None:
        print("   skipped: needs pygbx and replays/tmnf_exchange/Hockolicious")
        return
    results, stats = fake
    pkg, constraint = _constraint()
    score = score_rollout(constraint, results, stats, pkg.refs[0])
    n = n_decisions(results)
    tr = rollout_to_transitions(results, score, lambda_s=0.0, lambda_f=0.0, n_steps=3, gamma=0.99, reward_scale=1.0)
    assert tr["img"].shape == (n, 1, *HW) and tr["floats"].shape == (n, FLOAT_DIM) and tr["action"].shape == (n, 3)
    assert tr["next_img"].shape == tr["img"].shape and tr["reward_n"].shape == (n,) == tr["discount_n"].shape
    assert np.allclose(tr["discount_n"][:-3], 0.99**3) and np.all(tr["discount_n"][-3:] == 0.0)  # the last 3 reach the finish
    assert np.array_equal(tr["next_img"][0], tr["img"][3]) and np.array_equal(tr["next_floats"][5], tr["floats"][8])
    # a heavier style multiplier lowers the reward exactly where the corner costs sit
    heavy = rollout_to_transitions(results, score, lambda_s=2.0, lambda_f=0.0, n_steps=3, gamma=0.99, reward_scale=1.0)
    lowered = np.flatnonzero(heavy["reward_n"] < tr["reward_n"] - 1e-6)
    assert len(lowered) > 0 and score.style_total > 0
    corner_steps = np.flatnonzero(score.c_style > 0) - 1  # state index i receives the reward arriving in i + 1 ... i + 3
    assert set(lowered) <= set(np.concatenate([corner_steps - 2, corner_steps - 1, corner_steps]).tolist())
    assert np.isclose((tr["reward_n"].sum() - heavy["reward_n"].sum()) / 2.0, _discounted_cost_mass(score, 3, 0.99), rtol=1e-3)
    scaled = rollout_to_transitions(results, score, 0.0, 0.0, 3, 0.99, reward_scale=10.0)
    assert np.allclose(scaled["reward_n"], 10 * tr["reward_n"], rtol=1e-5)

    # cut short (not finished): bootstraps from the last state, one transition fewer than states
    cut = dict(results, frames=results["frames"][:-1])
    cut_score = score_rollout(constraint, cut, dict(race_finished=False), pkg.refs[0])
    tc = rollout_to_transitions(cut, cut_score, 0.1, 0.5, 3, 0.99)
    assert len(tc["reward_n"]) == n - 1 and np.all(tc["discount_n"] > 0)


def _discounted_cost_mass(score, n_steps, gamma):
    """Sum over the n-step windows of the discounted style cost: what lambda_s multiplies in the sum of reward_n."""
    T = len(score.c_style)
    total = 0.0
    for i in range(T - 1):
        m = min(n_steps, T - 1 - i)
        total += sum(gamma**j * score.c_style[i + 1 + j] for j in range(m))
    return total


def _cfg(**over):
    cfg = dict(
        sac_float_input_dim=FLOAT_DIM, sac_float_hidden_dim=32, sac_hidden_dim=64, sac_init_alpha=0.1, sac_lr_critic=3e-4, sac_lr_actor=3e-4, sac_lr_alpha=3e-4,
        sac_tau=0.005, sac_target_entropy=-3.0, sac_replay_capacity=2500, sac_lambda_s_init=0.1, sac_lambda_f_init=0.5, sac_eta_s=0.1, sac_eta_f=0.1,
        sac_style_budget=2.0, sac_failure_budget=0.1, sac_n_steps=3, sac_gamma=0.99, sac_reward_scale=10.0, sac_lambda_update_every_laps=2,
        sac_target_gain_ms=300, sac_start_learning_after=100, sac_batch_size=16, sac_updates_per_transition=0.01,
    )  # fmt: skip
    cfg.update(over)
    return SimpleNamespace(**cfg)


def test_learner_end_to_end_on_replayed_laps():
    fake = _fake_rollout()
    if fake is None:
        print("   skipped: needs pygbx and replays/tmnf_exchange/Hockolicious")
        return
    results, stats = fake
    pkg, constraint = _constraint()
    score = score_rollout(constraint, results, stats, pkg.refs[0])
    cfg = _cfg()
    learner = SACLearner(cfg, np.zeros(FLOAT_DIM), np.ones(FLOAT_DIM), HW, best_time_ms=54040 + 1000, device="cpu")

    assert learner.updates_due() == 0  # nothing in the replay yet
    rec = learner.add_rollout(results, stats, score, fill_buffer=True, is_explo=True, map_name="hock")
    n = n_decisions(results)
    assert len(learner.buffer) == n and learner.stats["transitions_added"] == n and learner.stats["steps_played"] == n
    assert rec["finished"] and not rec["failed"] and rec["new_fastest"] and rec["new_target"]  # 54040 <= 54040 + 1000 - 300 and style within budget
    rec2 = learner.add_rollout(results, stats, score, fill_buffer=True, is_explo=False, map_name="hock")
    assert not rec2["new_fastest"] and not rec2["new_target"]  # not faster than the one before
    assert learner.last_multiplier_update is not None and learner.pending_scores == []  # two laps: multipliers moved
    # the laps are valid and cost about 0.8 < B = 2: lambda_s = max(0, 0.1 + 0.1 * (0.8 - 2)) = 0 and lambda_f = 0.5 + 0.1 * (0 - 0.1) = 0.49
    assert learner.multipliers.lambda_s == 0.0 and np.isclose(learner.multipliers.lambda_f, 0.49)

    eval_only = learner.add_rollout(results, stats, score, fill_buffer=False, is_explo=False, map_name="hock")
    assert len(learner.buffer) == min(2 * n, cfg.sac_replay_capacity) and eval_only["lap"] == 3  # evaluation laps are not stored

    due = learner.updates_due()
    assert due == int(2 * n * cfg.sac_updates_per_transition) > 5
    metrics = learner.train(5)
    assert all(np.isfinite(v) for v in metrics.values()) and learner.stats["updates"] == 5
    assert learner.updates_due() == due - 5
    state = learner.actor_state()
    assert all(not v.requires_grad for v in state.values())

    # a failed lap (cut short) is not a target and not the fastest, and counts towards lambda_f
    bad_score = score_rollout(constraint, dict(results, frames=results["frames"][:-1]), dict(race_finished=False), pkg.refs[0])
    rec3 = learner.add_rollout(results, dict(race_finished=False), bad_score, True, True, "hock")
    assert rec3["failed"] and not rec3["new_fastest"] and not rec3["new_target"] and rec3["lap_time_ms"] is None

    with tempfile.TemporaryDirectory() as tmp:
        learner.save(Path(tmp))
        other = SACLearner(cfg, np.zeros(FLOAT_DIM), np.ones(FLOAT_DIM), HW, best_time_ms=54040 + 1000, device="cpu", seed=5)
        assert other.load(Path(tmp)) and other.stats == learner.stats and other.multipliers.lambda_s == learner.multipliers.lambda_s
        assert other.updates_due() == 0  # the replay is not saved: the update-to-data ratio starts again with this session
        for a, b in zip(other.net.parameters(), learner.net.parameters()):
            assert np.array_equal(a.detach().numpy(), b.detach().numpy())
        empty = SACLearner(cfg, np.zeros(FLOAT_DIM), np.ones(FLOAT_DIM), HW, best_time_ms=None, device="cpu")
        assert not empty.load(Path(tmp) / "nothing")


def test_exploration_policy_warms_up_then_follows_the_actor():
    torch_net = SACNet(FLOAT_DIM, np.zeros(FLOAT_DIM), np.ones(FLOAT_DIM), float_hidden_dim=32, hidden_dim=64, img_hw=HW)
    policy = SACExplorationPolicy(SACInferer(torch_net.actor, "cpu"), random_steps=3, seed=1)
    img, floats = np.zeros((1, *HW), np.uint8), np.zeros(FLOAT_DIM, np.float32)
    for _ in range(3):
        a, greedy, value, q = policy(img, floats)
        assert a[1] == 1.0 and a[2] == -1.0 and abs(a[0]) < 1 and not greedy  # raw throttle +1 -> full throttle, raw brake -1 -> no brake
    a, greedy, _, _ = policy(img, floats)  # warm-up is over: the actor
    assert a.shape == (3,) and np.all(np.abs(a) < 1) and not greedy
    policy.explore = False
    a1, greedy, _, _ = policy(img, floats)
    a2, _, _, _ = policy(img, floats)
    assert greedy and np.array_equal(a1, a2)  # the mean action is deterministic
    policy.explore, policy.random_steps = False, 5
    assert policy(img, floats)[1] is True  # evaluation workers never take the warm-up


def test_timelines_are_written_as_a_tmi_inputs_file():
    brake = np.array([0, 0, 1, 1, 0, 0, 0, 0], bool)
    accel = np.array([1, 1, 1, 1, 1, 0, 0, 1], bool)
    steer = np.array([0.0, 0.0, 0.5, 0.5, -1.0, -1.0, 0.0, 0.0])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.inputs"
        run_to_video.write_timelines_in_tmi_format(brake, accel, steer, path)
        lines = path.read_text().split("\n")
    assert "0.02 steer 32768" in lines and "0.04 steer -65536" in lines and "0.06 steer 0" in lines
    assert "0.0-0.05 press up" in lines and "0.07-0.08 press up" in lines and "0.02-0.04 press down" in lines


def test_reference_features_are_part_of_the_state_size_in_the_config():
    from config_files import config

    assert config.sac_n_reference_features == N_REFERENCE_FEATURES
    assert config.sac_float_input_dim == config.float_input_dim + N_REFERENCE_FEATURES
    assert len(config.float_inputs_mean) == len(config.float_inputs_std) == config.float_input_dim


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
