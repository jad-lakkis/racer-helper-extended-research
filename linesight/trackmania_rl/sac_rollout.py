"""
Glue between a rollout of the game (GameInstanceManager.rollout in continuous mode) and the SAC learner:

    previous_action_features   the "previous actions" part of the state, from the executed continuous actions
    SACExplorationPolicy       what the rollout calls on every decision
    score_rollout              style cost / failure cost / progress reward of a finished or cut rollout (StyleConstraint)
    rollout_to_transitions     rollout + lambdas -> n-step transitions for the replay buffer

No game and no config are needed here, so all of it is tested offline (tests/test_sac_pipeline.py).
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from trackmania_rl.agents.sac import SACInferer, make_nstep_transitions
from trackmania_rl.style_constraint import LapScore, StyleConstraint

DEFAULT_ACTION = np.array([0.0, 1.0, 0.0])  # (steer, throttle, brake) assumed before the first decision: straight, full throttle


def previous_action_features(executed_actions: Sequence[np.ndarray], n_prev: int) -> list:
    """
    Four numbers per previous action, in the order of the discrete state (accelerate, brake, amount of left steering, amount of right steering),
    for the n_prev latest executed (steer, throttle, brake) actions; missing ones count as DEFAULT_ACTION.
    """
    values = []
    for k in range(len(executed_actions) - n_prev, len(executed_actions)):
        steer, throttle, brake = executed_actions[k] if k >= 0 else DEFAULT_ACTION
        values += [throttle, brake, max(0.0, -steer), max(0.0, steer)]
    return values


class SACExplorationPolicy:
    """
    exploration_policy for GameInstanceManager.rollout in continuous mode: (frame, floats) -> (raw action, is_greedy, value, q_values).
    The raw action is the network output in [-1, 1]^3 (the replay stores it); the rollout maps it to (steer, throttle, brake).
    While `explore` is set the action is sampled from the policy, otherwise it is the mean. The first `random_steps` decisions of an
    exploring worker are a wobbling full-throttle drive instead (an untrained policy mostly stands still or crashes).
    """

    def __init__(self, inferer: SACInferer, random_steps: int = 0, seed: int = 0):
        self.inferer = inferer
        self.explore = True
        self.random_steps = random_steps
        self.rng = np.random.default_rng(seed)

    def __call__(self, frame: np.ndarray, floats: np.ndarray) -> Tuple[np.ndarray, bool, float, np.ndarray]:
        if self.explore and self.random_steps > 0:
            self.random_steps -= 1
            action = np.array([np.clip(self.rng.normal(0.0, 0.3), -1, 1), 1.0, -1.0], dtype=np.float32)  # steer wobble, throttle on, brake off
            return action, False, 0.0, np.zeros(1, dtype=np.float32)
        action = self.inferer.act(frame, floats, explore=self.explore)
        return action, not self.explore, 0.0, np.zeros(1, dtype=np.float32)


def n_decisions(rollout_results: dict) -> int:
    """Number of decisions of a rollout that have a state, an image and an executed action."""
    return min(len(rollout_results["positions"]), len(rollout_results["executed_actions"]))


def score_rollout(constraint: StyleConstraint, rollout_results: dict, end_race_stats: dict, reference) -> Optional[LapScore]:
    """Reward and costs of a rollout (None if it is too short to say anything). reference: the ReferenceLap that was in the state."""
    n = n_decisions(rollout_results)
    if n < 3:
        return None
    finished = bool(end_race_stats.get("race_finished", False))
    positions = np.asarray(rollout_results["positions"][:n], dtype=np.float64)
    speeds = np.asarray(rollout_results["speeds_kmh"][:n], dtype=np.float64)
    actions = np.asarray(rollout_results["executed_actions"][:n], dtype=np.float64)
    race_time_ms = None
    if finished and rollout_results.get("end_position") is not None:
        positions = np.vstack([positions, np.asarray(rollout_results["end_position"], dtype=np.float64)[None]])
        speeds = np.append(speeds, speeds[-1])
        race_time_ms = int(end_race_stats["race_time"])
    else:
        finished = False
    return constraint.score(positions, speeds, actions, finished, race_time_ms, reference)


def rollout_to_transitions(
    rollout_results: dict, score: LapScore, lambda_s: float, lambda_f: float, n_steps: int, gamma: float, reward_scale: float = 1.0
) -> Optional[dict]:
    """
    n-step transitions of a rollout with the adjusted reward  r - lambda_s * c_style - lambda_f * c_fail.
    A rollout that finished ends in a terminal state with no image (no bootstrap); one that was cut short bootstraps from its last state.
    """
    n = n_decisions(rollout_results)
    if n < 3 or score is None:
        return None
    finished = score.lap_time_ms is not None
    imgs = np.stack(rollout_results["frames"][:n])  # (n, 1, H, W) uint8
    floats = np.stack(rollout_results["state_float"][:n]).astype(np.float32)
    actions = np.stack(rollout_results["actions"][:n]).astype(np.float32)
    if finished:  # the terminal state has no image: repeat the last one (its discount is 0, it is never used)
        imgs = np.concatenate([imgs, imgs[-1:]])
        floats = np.concatenate([floats, floats[-1:]])
    adjusted = reward_scale * score.adjusted_reward(lambda_s, lambda_f)
    assert len(adjusted) == len(imgs), (len(adjusted), len(imgs))
    nxt, reward_n, discount_n = make_nstep_transitions(adjusted, n_steps, gamma, ended_in_finish=finished)
    last = len(nxt)
    return dict(
        img=imgs[:last],
        floats=floats[:last],
        action=actions[:last],
        reward_n=reward_n,
        discount_n=discount_n,
        next_img=imgs[nxt],
        next_floats=floats[nxt],
    )
