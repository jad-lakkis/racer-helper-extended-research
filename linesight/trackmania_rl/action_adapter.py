"""
Continuous actions for Trackmania Nations Forever.

The agent chooses three numbers every decision (40 ms = 4 physics ticks of 10 ms in the default setting), the same three that
Assetto Corsa takes:

    steer    in [-1, 1]   negative = left      sent to the game as an analog value in [-65536, 65536]
    throttle in [ 0, 1]
    brake    in [ 0, 1]

TMNF has no analog pedals: a pedal is down or up on each 10 ms tick. A pedal level k / ticks is therefore played as "down" on
k of the `ticks` ticks of the decision, spread evenly and starting with a tick that is down (duty cycle). With 4 ticks the levels are
0, 0.25, 0.5, 0.75 and 1. Whether the car answers such a pattern like a partial pedal still has to be measured in the game
(scripts/tools/pedal_response_test.py).

This module has no dependency on the game or on config, so it can be tested on its own.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    ticks_per_action: int = 4  # physics ticks (10 ms each) per decision; 4 = 40 ms, as in the AssettoCorsaGym benchmark
    steer_step: float = 0.0  # 0: steering is sent as the agent gives it; 0.1 or 0.2: rounded to that grid

    def __post_init__(self):
        if not 1 <= self.ticks_per_action <= 16:
            raise ValueError("ticks_per_action must be between 1 and 16")
        if self.steer_step < 0 or self.steer_step > 1:
            raise ValueError("steer_step must be in [0, 1]")

    @property
    def decision_ms(self) -> int:
        return 10 * self.ticks_per_action

    @property
    def pedal_levels(self) -> np.ndarray:
        return np.arange(self.ticks_per_action + 1) / self.ticks_per_action


def from_policy_output(raw: np.ndarray) -> np.ndarray:
    """Map a network output in [-1, 1]^3 (e.g. after tanh) to (steer in [-1, 1], throttle in [0, 1], brake in [0, 1])."""
    raw = np.clip(np.asarray(raw, dtype=np.float64), -1.0, 1.0)
    return np.array([raw[0], (raw[1] + 1.0) / 2.0, (raw[2] + 1.0) / 2.0])


def to_policy_output(action: np.ndarray) -> np.ndarray:
    """Inverse of from_policy_output."""
    a = np.asarray(action, dtype=np.float64)
    return np.array([a[0], 2.0 * a[1] - 1.0, 2.0 * a[2] - 1.0])


def quantize(action: np.ndarray, spec: ActionSpec) -> np.ndarray:
    """The action that is actually played: steering rounded to steer_step (if set), pedals rounded to a multiple of 1 / ticks_per_action."""
    steer, throttle, brake = (float(x) for x in action)
    steer = float(np.clip(steer, -1.0, 1.0))
    if spec.steer_step > 0:
        steer = float(np.clip(np.round(steer / spec.steer_step) * spec.steer_step, -1.0, 1.0))
    k = spec.ticks_per_action
    throttle = float(np.round(np.clip(throttle, 0.0, 1.0) * k) / k)
    brake = float(np.round(np.clip(brake, 0.0, 1.0) * k) / k)
    return np.array([steer, throttle, brake])


def duty_pattern(level: float, ticks: int) -> np.ndarray:
    """
    Which of the `ticks` ticks have the pedal down for a pedal level in [0, 1] (a multiple of 1 / ticks): the pedal is down on
    round(level * ticks) ticks, spread evenly, with tick 0 down whenever the level is above 0.
        level 0.5, 4 ticks -> [1, 0, 1, 0]      level 0.25 -> [1, 0, 0, 0]      level 0.75 -> [1, 1, 1, 0]
    """
    k = int(round(float(np.clip(level, 0.0, 1.0)) * ticks))
    i = np.arange(ticks)
    return np.ceil((i + 1) * k / ticks - 1e-9) > np.ceil(i * k / ticks - 1e-9)


def pattern_to_mask(pattern: np.ndarray) -> int:
    """Bit i of the mask is set when the pedal is down on tick i."""
    return int(sum(1 << i for i, on in enumerate(pattern) if on))


def mask_to_pattern(mask: int, ticks: int) -> np.ndarray:
    return np.array([(mask >> i) & 1 for i in range(ticks)], dtype=bool)


def game_command(action: np.ndarray, spec: ActionSpec) -> Tuple[int, int, int]:
    """(steer integer in [-65536, 65536], throttle mask, brake mask) to send to the game for one decision."""
    steer, throttle, brake = quantize(action, spec)
    k = spec.ticks_per_action
    return (
        int(round(steer * 65536)),
        pattern_to_mask(duty_pattern(throttle, k)),
        pattern_to_mask(duty_pattern(brake, k)),
    )


def input_timelines(actions: np.ndarray, spec: ActionSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The inputs of a whole rollout on the 10 ms grid, as the style features expect them.
    actions: (n, 3) executed actions (steer, throttle, brake). Returns (brake bool, accel bool, steer float), each of length n * ticks.
    """
    k = spec.ticks_per_action
    brake = np.zeros(len(actions) * k, dtype=bool)
    accel = np.zeros(len(actions) * k, dtype=bool)
    steer = np.zeros(len(actions) * k)
    for i, a in enumerate(actions):
        q = quantize(a, spec)
        steer[i * k : (i + 1) * k] = q[0]
        accel[i * k : (i + 1) * k] = duty_pattern(q[1], k)
        brake[i * k : (i + 1) * k] = duty_pattern(q[2], k)
    return brake, accel, steer
