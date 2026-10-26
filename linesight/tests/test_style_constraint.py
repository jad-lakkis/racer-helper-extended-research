"""
Offline checks for trackmania_rl/style_constraint.py. Run from linesight/:  python tests/test_style_constraint.py

The unit checks need only numpy. The replay check also needs pygbx and the Hockolicious replays under replays/tmnf_exchange (it is skipped
with a message if they are missing). It plays a human lap through the path an agent's rollout takes (a position, a speed and a
(steer, throttle, brake) action every 40 ms, pedals as 4-tick duty cycles) and compares the style numbers with the ones the extractor
wrote to style_vectors/Hockolicious/corners.csv for the same replay.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "linesight"))
sys.path.insert(0, str(ROOT / "linesight" / "scripts"))
from trackmania_rl.action_adapter import ActionSpec  # noqa: E402
from trackmania_rl.style_constraint import (  # noqa: E402
    LapScore,
    Multipliers,
    StyleConstraint,
    corner_style_cost,
    select_target,
)

REPLAY = ROOT / "replays" / "tmnf_exchange" / "Hockolicious" / "ESL-Hockolicious_RtA-wally_54040_6143956.Replay.Gbx"
STYLE_DIR = ROOT / "style_vectors" / "Hockolicious"
DRIVER, LAP_ID = "-wally-", "Hockolicious_-wally-_54040"


def test_cost_is_zero_inside_the_ranges_and_grows_outside():
    ranges = {"b": (0.2, 0.6), "f": (0.0, 0.5), "l_apex": (0.1, 0.3), "s": (1.0, 3.0)}
    inside = dict(b=0.2, f=0.25, l_apex=0.3, s=2.0)  # on the edges counts as inside
    assert corner_style_cost(inside, ranges)[0] == 0.0
    cost, per = corner_style_cost(dict(b=0.8, f=0.25, l_apex=0.0, s=2.0), ranges)
    assert np.isclose(per["b"], 0.2 / 0.4) and np.isclose(per["l_apex"], 0.1 / 0.2) and per["f"] == 0 and per["s"] == 0
    assert np.isclose(cost, per["b"] + per["l_apex"])
    # further away costs more, up to the cap
    assert corner_style_cost(dict(b=1.0), ranges)[0] > corner_style_cost(dict(b=0.8), ranges)[0]
    assert corner_style_cost(dict(b=50.0), ranges)[0] == 3.0


def test_narrow_ranges_use_the_width_floor_and_missing_values_cost_nothing():
    # all laps had exactly b = 0 in this corner: leaving by 0.05 is half a floor-width (0.10), not an infinite cost
    cost, per = corner_style_cost(dict(b=0.05), {"b": (0.0, 0.0)})
    assert np.isclose(per["b"], 0.5)
    assert corner_style_cost(dict(b=float("nan"), f=None), {"b": (0.0, 0.0), "f": (0.0, 1.0)})[0] == 0.0
    assert corner_style_cost(dict(b=0.9), {})[0] == 0.0  # no range known for this corner


def test_multipliers_move_towards_the_constraints():
    def lap(style, failed):
        return LapScore(np.zeros(3), np.zeros(3), np.zeros(3), style, failed, "", None, [], {}, 0.0)

    m = Multipliers(lambda_s=1.0, lambda_f=1.0, eta_s=0.1, eta_f=0.5, budget_style=2.0, budget_fail=0.1)
    out = m.update([lap(6.0, True), lap(4.0, False)])  # mean style 5 > B = 2, failure rate 0.5 > eps = 0.1
    assert np.isclose(out["lambda_s"], 1.0 + 0.1 * (5.0 - 2.0)) and np.isclose(out["lambda_f"], 1.0 + 0.5 * (0.5 - 0.1))
    for _ in range(100):  # constraints satisfied: the multipliers decay to 0 and stay there (max(0, .))
        out = m.update([lap(0.0, False), lap(1.0, False)])
    assert out["lambda_s"] == 0.0 and out["lambda_f"] == 0.0
    big = Multipliers(lambda_s=99.9, eta_s=10.0, max_lambda=100.0)
    assert big.update([lap(100.0, False)])["lambda_s"] == 100.0


def test_adjusted_reward_follows_the_formulation():
    s = LapScore(np.array([0.0, 1.0, 1.0, 1.0]), np.array([0.0, 0.0, 3.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]), 3.0, True, "x", None, [], {}, 0.0)
    assert np.allclose(s.adjusted_reward(0.5, 2.0), [0.0, 1.0, 1.0 - 1.5, 1.0 - 2.0])


def test_select_target():
    laps = [
        dict(lap_time_ms=53000, style_total=5.0, failed=False),  # fast but too far from the style
        dict(lap_time_ms=53200, style_total=1.0, failed=False),  # the one
        dict(lap_time_ms=53400, style_total=0.5, failed=False),  # also qualifies but slower
        dict(lap_time_ms=52000, style_total=0.0, failed=True),  # failed laps never count
        dict(lap_time_ms=None, style_total=0.0, failed=True),
        dict(lap_time_ms=53900, style_total=0.0, failed=False),  # not enough faster than the best
    ]
    assert select_target(laps, best_time_ms=54040, gain_ms=500, style_budget=2.0)["lap_time_ms"] == 53200
    assert select_target(laps, best_time_ms=54040, gain_ms=2000, style_budget=2.0) is None


# ─────────────────────────────────────────────────────────────────────────────
# The replay: a human lap seen the way an agent's rollout is
# ─────────────────────────────────────────────────────────────────────────────


def _human_lap_as_a_rollout():
    """Positions, speeds and (steer, throttle, brake) per 40 ms decision of the wally lap; None if pygbx or the files are missing."""
    try:
        from extract_driver_profile import _load_ghost
        from extract_style_vectors import _steer_signal, _timeline
    except SystemExit:
        return None
    except ImportError:
        return None
    if not (REPLAY.exists() and STYLE_DIR.exists()):
        return None
    ghost, race_start, race_time, n_samples = _load_ghost(str(REPLAY))
    recs = ghost.records[: n_samples + 1]
    pos100 = np.array([[r.position.x, r.position.y, r.position.z] for r in recs])
    speed100 = np.array([r.display_speed for r in recs], dtype=float)
    t100 = np.minimum(np.arange(len(recs)) * 0.1, race_time / 1000.0)
    n10 = race_time // 10 + 1
    ce = ghost.control_entries
    brake, accel = _timeline(ce, "Brake", race_start, n10), _timeline(ce, "Accelerate", race_start, n10)
    u, _ = _steer_signal(ce, race_start, n10)

    n_dec = race_time // 40
    t = np.arange(n_dec + 1) * 0.04
    t[-1] = race_time / 1000.0  # the last state is the finish
    pos = np.stack([np.interp(t, t100, pos100[:, i]) for i in range(3)], axis=1)
    speed = np.interp(t, t100, speed100)
    per = lambda a: np.array([a[4 * i : 4 * i + 4].mean() for i in range(n_dec)])  # noqa: E731  (mean over the 4 ticks of a decision)
    actions = np.stack([per(u), per(accel.astype(float)), per(brake.astype(float))], axis=1)
    return pos, speed, actions, int(race_time)


def _constraint():
    from trackmania_rl.style_constraint import DriverStyle

    return StyleConstraint(DriverStyle.from_style_vectors(STYLE_DIR, DRIVER), ActionSpec(ticks_per_action=4))


def test_agent_path_reproduces_the_extractor_on_a_human_replay():
    lap = _human_lap_as_a_rollout()
    if lap is None:
        print("   skipped: needs pygbx and replays/tmnf_exchange/Hockolicious")
        return
    import pandas as pd

    pos, speed, actions, race_time = lap
    score = _constraint().score(pos, speed, actions, finished=True, race_time_ms=race_time)
    assert not score.failed and score.lap_time_ms == 54040 and score.progress_m > 0.99 * 4019, (score.failed, score.fail_reason)

    ref = pd.read_csv(STYLE_DIR / "corners.csv")
    ref = ref[ref["lap_id"] == LAP_ID].set_index("corner")
    got = pd.DataFrame(score.rows).set_index("corner")
    assert list(got.index) == list(ref.index)
    for m, tol in dict(b=0.05, f=0.05, l_entry=0.03, l_apex=0.04, l_exit=0.03, s=0.6).items():
        worst = float((got[m] - ref[m]).abs().max())
        print(f"   {m:8s} max |agent path - extractor| = {worst:.3f}  (tolerance {tol})")
        assert worst < tol, (m, worst)

    # the lap is one of the two that define this driver's ranges, so it costs (nearly) nothing: what is left is the 40 ms resampling
    print(f"   style cost of the driver's own lap: {score.style_total:.2f}")
    assert score.style_total < 2.0


def test_bad_and_unfinished_laps_are_penalised():
    lap = _human_lap_as_a_rollout()
    if lap is None:
        print("   skipped: needs pygbx and replays/tmnf_exchange/Hockolicious")
        return
    pos, speed, actions, race_time = lap
    sc = _constraint()
    own = sc.score(pos, speed, actions, finished=True, race_time_ms=race_time)

    never_brakes = actions.copy()
    never_brakes[:, 1], never_brakes[:, 2] = 1.0, 0.0  # flat out everywhere: the lap's positions are unchanged, only the pedals differ
    flat = sc.score(pos, speed, never_brakes, finished=True, race_time_ms=race_time)
    print(f"   style cost: own lap {own.style_total:.2f}, flat out with no brake {flat.style_total:.2f}")
    assert flat.style_total > own.style_total + 5.0

    half = len(pos) // 2
    cut = sc.score(pos[:half], speed[:half], actions[: half - 1], finished=False)
    assert cut.failed and cut.fail_reason == "not finished" and cut.lap_time_ms is None
    assert cut.c_fail[-1] == 1.0 and cut.c_fail[:-1].sum() == 0.0
    adj = cut.adjusted_reward(lambda_s=0.1, lambda_f=3.0)
    assert np.isclose(adj[-1], cut.base_reward[-1] - 3.0 - 0.1 * cut.c_style[-1])

    # a lap that is run on the wrong side of the world (off the road all the way) is a failure even if it "finishes"
    off = pos.copy()
    off[:, 0] += 40.0
    far = sc.score(off, speed, actions, finished=True, race_time_ms=race_time)
    assert far.failed, far.fail_reason


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
