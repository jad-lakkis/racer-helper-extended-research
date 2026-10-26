"""
This file's main entry point is the function fill_buffer_from_rollout_with_n_steps_rule().
Its main inputs are a rollout_results object (obtained from a GameInstanceManager object), and a buffer to be filled.
It reassembles the rollout_results object into transitions, as defined in /trackmania_rl/experience_replay/experience_replay_interface.py
"""

import math
import random

import numpy as np
from numba import jit
from torchrl.data import ReplayBuffer

from config_files import config_copy
from trackmania_rl.experience_replay.experience_replay_interface import Experience
from trackmania_rl.reward_shaping import speedslide_quality_tarmac


@jit(nopython=True)
def get_potential(state_float):
    # https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    vector_vcp_to_vcp_further_ahead = state_float[65:68] - state_float[62:65]
    vector_vcp_to_vcp_further_ahead_normalized = vector_vcp_to_vcp_further_ahead / np.linalg.norm(vector_vcp_to_vcp_further_ahead)

    return (
        config_copy.shaped_reward_dist_to_cur_vcp
        * max(
            config_copy.shaped_reward_min_dist_to_cur_vcp,
            min(config_copy.shaped_reward_max_dist_to_cur_vcp, np.linalg.norm(state_float[62:65])),
        )
    ) + (config_copy.shaped_reward_point_to_vcp_ahead * (vector_vcp_to_vcp_further_ahead_normalized[2] - 1))


def fill_buffer_from_rollout_with_n_steps_rule(
    buffer: ReplayBuffer,
    buffer_test: ReplayBuffer,
    rollout_results: dict,
    n_steps_max: int,
    gamma: float,
    discard_non_greedy_actions_in_nsteps: bool,
    engineered_speedslide_reward: float,
    engineered_neoslide_reward: float,
    engineered_kamikaze_reward: float,
    engineered_close_to_vcp_reward: float,
    humanlike_oscillation_penalty: float,
    humanlike_brake_tap_penalty: float,
    humanlike_low_speed_slide_penalty: float,
    humanlike_braking_aggression_reward: float,
    braking_aggression: float,
    humanlike_risk_tolerance_reward: float,
    risk_tolerance: float,
    humanlike_oversteer_understeer_reward: float,
    oversteer_understeer_score: float,
    humanlike_steer_tap_penalty: float,
    humanlike_accel_tap_penalty: float,
    humanlike_corner_entry_speed_reward: float,
    corner_entry_speed_ratio: float,
):
    assert len(rollout_results["frames"]) == len(rollout_results["current_zone_idx"])
    n_frames = len(rollout_results["frames"])

    number_memories_added_train = 0
    number_memories_added_test = 0
    Experiences_For_Buffer = []
    Experiences_For_Buffer_Test = []
    list_to_fill = Experiences_For_Buffer_Test if random.random() < config_copy.buffer_test_ratio else Experiences_For_Buffer

    gammas = (gamma ** np.linspace(1, n_steps_max, n_steps_max)).astype(
        np.float32
    )  # Discount factor that will be placed in front of next_step in Bellman equation, depending on n_steps chosen

    reward_into = np.zeros(n_frames)

    # =========================================================
    # Human-likeness pre-passes (run before the main reward loop)
    # =========================================================

    # --- Brake tap penalty ---
    # A brake press held for fewer than MIN_BRAKE_HOLD_STEPS consecutive steps (150 ms) is not
    # human-like. We detect the moment of release and tag that slot; the penalty is then added
    # inside the main loop below.
    MIN_BRAKE_HOLD_STEPS = 3  # 150 ms at 50 ms/step
    brake_tap_penalty_at = np.zeros(n_frames)
    if humanlike_brake_tap_penalty != 0:
        brake_hold = 0
        for _idx in range(n_frames - 1):  # n_frames-1 excludes terminal nan frame
            _a = config_copy.inputs[int(rollout_results["actions"][_idx])]
            if _a["brake"]:
                brake_hold += 1
            else:
                if 0 < brake_hold < MIN_BRAKE_HOLD_STEPS:
                    # Release step: reward_into[_idx] corresponds to Experience _idx-1
                    # (the last brake step) — tagging here attributes the penalty correctly
                    brake_tap_penalty_at[_idx] = humanlike_brake_tap_penalty
                brake_hold = 0

    # --- Steering tap penalty ---
    # A left or right press held fewer than MIN_STEER_HOLD_STEPS (150 ms) is a micro-tap.
    # Detected at the release step, same accounting as brake tap.
    # Left and right holds are tracked independently so a direct L→R transition
    # correctly triggers the tap check for left at the step right begins.
    MIN_STEER_HOLD_STEPS = 3  # 150 ms at 50 ms/step
    steer_tap_penalty_at = np.zeros(n_frames)
    if humanlike_steer_tap_penalty != 0:
        left_hold = 0
        right_hold = 0
        for _idx in range(n_frames - 1):
            _a = config_copy.inputs[int(rollout_results["actions"][_idx])]
            if not _a["left"] and left_hold > 0:
                if left_hold < MIN_STEER_HOLD_STEPS:
                    steer_tap_penalty_at[_idx] = humanlike_steer_tap_penalty
                left_hold = 0
            if not _a["right"] and right_hold > 0:
                if right_hold < MIN_STEER_HOLD_STEPS:
                    steer_tap_penalty_at[_idx] = humanlike_steer_tap_penalty
                right_hold = 0
            if _a["left"]:
                left_hold += 1
            if _a["right"]:
                right_hold += 1

    # --- Accelerator tap penalty ---
    # A throttle press held fewer than MIN_ACCEL_HOLD_STEPS (150 ms) is a micro-tap —
    # no human driver blips the gas that briefly in a racing context.
    MIN_ACCEL_HOLD_STEPS = 3  # 150 ms at 50 ms/step
    accel_tap_penalty_at = np.zeros(n_frames)
    if humanlike_accel_tap_penalty != 0:
        accel_hold = 0
        for _idx in range(n_frames - 1):
            _a = config_copy.inputs[int(rollout_results["actions"][_idx])]
            if not _a["accelerate"] and accel_hold > 0:
                if accel_hold < MIN_ACCEL_HOLD_STEPS:
                    accel_tap_penalty_at[_idx] = humanlike_accel_tap_penalty
                accel_hold = 0
            if _a["accelerate"]:
                accel_hold += 1

    # --- Steering oscillation: prefix-sum of direction flips ---
    # A direction flip is a direct left↔right change (ignoring neutral steps).
    # We allow 1 flip per 200 ms window (natural cornering); every additional flip is penalised.
    STEER_OSCILLATION_WINDOW = 4  # 200 ms at 50 ms/step
    flip_cumsum = None
    if humanlike_oscillation_penalty != 0:
        flip_at_step = np.zeros(n_frames)
        last_nonzero_dir = 0
        for _idx in range(n_frames - 1):
            _a = config_copy.inputs[int(rollout_results["actions"][_idx])]
            _d = 1 if _a["left"] else (-1 if _a["right"] else 0)
            if _d != 0:
                if last_nonzero_dir != 0 and _d != last_nonzero_dir:
                    flip_at_step[_idx] = 1
                last_nonzero_dir = _d
        # Prepend a 0 so flip_cumsum[i] = sum(flip_at_step[0:i]), enabling O(1) window queries
        flip_cumsum = np.concatenate([[0.0], np.cumsum(flip_at_step)])

    # --- Corner entry speed alignment ---
    # Curvature proxy: κ = |yaw_rate| / max(|v_fwd|, 1.0)
    #   state_float index 54 = angular velocity Y (yaw rate, std≈1 rad/s)
    #   state_float index 58 = forward speed (mean≈55 m/s)
    # A rising edge above CORNER_CURV_THRESH marks a corner entry.
    # At each entry the Brier-score penalty fires:
    #   r = coeff × (entry_ratio − target_ratio)²
    # where entry_ratio = agent speed at entry / rollout peak speed.
    _CORNER_CURV_THRESH = 0.010  # 1/m — mirrors extract_driver_profile.py constant
    _CORNER_MIN_GAP = 10         # steps (500 ms) — matches extract_driver_profile.py (5 × 100 ms samples)
    _YAW_RATE_IDX = 54
    _V_FWD_IDX = 58
    corner_entry_speed_reward_at = np.zeros(n_frames)
    if humanlike_corner_entry_speed_reward != 0:
        kappa = np.array([
            abs(float(rollout_results["state_float"][_i][_YAW_RATE_IDX]))
            / max(abs(float(rollout_results["state_float"][_i][_V_FWD_IDX])), 1.0)
            for _i in range(n_frames - 1)
        ] + [0.0])  # terminal frame gets zero
        peak_speed = max(
            (abs(float(rollout_results["state_float"][_i][_V_FWD_IDX])) for _i in range(n_frames - 1)),
            default=1.0,
        )
        peak_speed = max(peak_speed, 1.0)
        above = kappa > _CORNER_CURV_THRESH
        last_entry = -_CORNER_MIN_GAP - 1
        for _i in range(1, n_frames - 1):
            if above[_i] and not above[_i - 1] and (_i - last_entry) >= _CORNER_MIN_GAP:
                entry_speed = abs(float(rollout_results["state_float"][max(0, _i - 1)][_V_FWD_IDX]))
                entry_ratio = entry_speed / peak_speed
                corner_entry_speed_reward_at[_i] = (
                    humanlike_corner_entry_speed_reward * (entry_ratio - corner_entry_speed_ratio) ** 2
                )
                last_entry = _i

    # Slip-ratio thresholds for oversteer/understeer detection
    _OVERSTEER_SLIP_SAT = 0.3   # |v_lat|/|v_fwd| at which oversteer signal saturates to 1
    _UNDERSTEER_SLIP_MAX = 0.04  # |v_lat|/|v_fwd| below which understeer fires when steering

    for i in range(1, n_frames):
        reward_into[i] += config_copy.constant_reward_per_ms * (
            config_copy.ms_per_action
            if (i < n_frames - 1 or ("race_time" not in rollout_results))
            else rollout_results["race_time"] - (n_frames - 2) * config_copy.ms_per_action
        )
        reward_into[i] += (
            rollout_results["meters_advanced_along_centerline"][i] - rollout_results["meters_advanced_along_centerline"][i - 1]
        ) * config_copy.reward_per_m_advanced_along_centerline
        if i < n_frames - 1:
            if config_copy.final_speed_reward_per_m_per_s != 0 and rollout_results["state_float"][i][58] > 0:
                # car has velocity *forward*
                reward_into[i] += config_copy.final_speed_reward_per_m_per_s * (
                    np.linalg.norm(rollout_results["state_float"][i][56:59]) - np.linalg.norm(rollout_results["state_float"][i - 1][56:59])
                )
            if engineered_speedslide_reward != 0 and np.all(rollout_results["state_float"][i][25:29]):
                # all wheels touch the ground
                reward_into[i] += engineered_speedslide_reward * max(
                    0.0,
                    1 - abs(speedslide_quality_tarmac(rollout_results["state_float"][i][56], rollout_results["state_float"][i][58]) - 1),
                )  # TODO : indices 25:29, 56 and 58 are hardcoded, this is bad....

            # lateral speed >= 2 m/s AND forward speed >= 10 m/s (36 km/h):
            # the forward-speed guard prevents overlap with humanlike_low_speed_slide_penalty,
            # which fires at forward speed < 10 m/s — without it the two terms cancel or conflict.
            reward_into[i] += (
                engineered_neoslide_reward
                if abs(rollout_results["state_float"][i][56]) >= 2.0
                and abs(rollout_results["state_float"][i][58]) >= 10.0
                else 0
            )  # TODO : 56, 58 are hardcoded, this is bad....
            # kamikaze reward
            if (
                engineered_kamikaze_reward != 0
                and rollout_results["actions"][i] < getattr(config_copy, "n_forward_actions", 3)
                or np.sum(rollout_results["state_float"][i][25:29]) <= 1
            ):
                reward_into[i] += engineered_kamikaze_reward
            if engineered_close_to_vcp_reward != 0:
                reward_into[i] += engineered_close_to_vcp_reward * max(
                    config_copy.engineered_reward_min_dist_to_cur_vcp,
                    min(config_copy.engineered_reward_max_dist_to_cur_vcp, np.linalg.norm(rollout_results["state_float"][i][62:65])),
                )

            # ---- Human-likeness penalties ----

            # Brake tap: short press detected in pre-pass; apply at the release step
            if humanlike_brake_tap_penalty != 0:
                reward_into[i] += brake_tap_penalty_at[i]

            # Steering tap: short left/right press detected in pre-pass; apply at release step
            if humanlike_steer_tap_penalty != 0:
                reward_into[i] += steer_tap_penalty_at[i]

            # Accelerator tap: short throttle press detected in pre-pass; apply at release step
            if humanlike_accel_tap_penalty != 0:
                reward_into[i] += accel_tap_penalty_at[i]

            # Steering oscillation: penalise every extra L↔R flip beyond 1 per 200 ms window
            # (1 flip = natural mid-corner correction; repeated flipping = inhuman tapping)
            if humanlike_oscillation_penalty != 0:
                window_flips = flip_cumsum[i] - flip_cumsum[max(0, i - STEER_OSCILLATION_WINDOW)]
                reward_into[i] += humanlike_oscillation_penalty * max(0.0, float(window_flips) - 1.0)

            # Neo slide at low speed: sliding wheels while forward speed < 36 km/h is an AI
            # artefact that no human driver produces intentionally at low speed
            if humanlike_low_speed_slide_penalty != 0:
                _LOW_SPEED_THRESHOLD = 10.0  # m/s ≈ 36 km/h
                _is_any_sliding = np.any(rollout_results["state_float"][i][21:25] > 0.5)
                _speed_fwd = abs(float(rollout_results["state_float"][i][58]))
                if _is_any_sliding and _speed_fwd < _LOW_SPEED_THRESHOLD:
                    reward_into[i] += humanlike_low_speed_slide_penalty

            # Braking aggression alignment — Brier-score penalty:
            #   r = coeff × (brake_t − α)²
            # This is the unique proper scoring rule for binary events: its expected
            # value is minimised exactly when P(brake|s) = braking_aggression.
            # coeff is negative, so this is always a penalty, maximally -|coeff| when
            # the agent's action is the polar opposite of the target (e.g. target=1
            # but agent does not brake → deviation² = 1).
            if humanlike_braking_aggression_reward != 0:
                _is_braking = float(config_copy.inputs[int(rollout_results["actions"][i])]["brake"])
                reward_into[i] += humanlike_braking_aggression_reward * (_is_braking - braking_aggression) ** 2

            # Risk tolerance alignment — Brier-score penalty on VCP lateral deviation:
            #   r = coeff × (dist_normalized − risk_tolerance)²
            # where dist_normalized = clip(||state_float[62:65]|| / dist_max, 0, 1).
            # dist_normalized ≈ 0: agent is on the centerline (conservative line).
            # dist_normalized ≈ 1: agent is dist_max metres from VCP (aggressive cut).
            # The Brier score drives dist_normalized → risk_tolerance at equilibrium:
            #   - risk_tolerance=0 → penalise any deviation from centerline
            #   - risk_tolerance=1 → penalise hugging the centerline (reward aggression)
            #   - risk_tolerance=0.5 → minimum penalty at ~7.5 m lateral deviation
            if humanlike_risk_tolerance_reward != 0:
                _dist_to_vcp = np.linalg.norm(rollout_results["state_float"][i][62:65])
                _dist_normalized = min(_dist_to_vcp / config_copy.risk_tolerance_vcp_dist_max, 1.0)
                reward_into[i] += humanlike_risk_tolerance_reward * (_dist_normalized - risk_tolerance) ** 2

            # Oversteer / understeer alignment:
            #   o_signal ∈ [-1,1]: +1=oversteering (high lateral slip), -1=understeering
            #   (steering applied but near-zero lateral response)
            #   reward = coeff × (score/5) × o_signal → positive when style matches score
            if humanlike_oversteer_understeer_reward != 0 and oversteer_understeer_score != 0:
                _v_fwd = abs(float(rollout_results["state_float"][i][58]))
                if _v_fwd >= 10.0:  # skip at low speed
                    _v_lat = abs(float(rollout_results["state_float"][i][56]))
                    _slip = min(_v_lat / max(_v_fwd, 1.0), _OVERSTEER_SLIP_SAT) / _OVERSTEER_SLIP_SAT
                    _action = config_copy.inputs[int(rollout_results["actions"][i])]
                    _is_steering = _action["left"] or _action["right"]
                    _is_understeering = _is_steering and (_v_lat / max(_v_fwd, 1.0) < _UNDERSTEER_SLIP_MAX)
                    _o_signal = max(-1.0, min(1.0, _slip - (1.0 if _is_understeering else 0.0)))
                    reward_into[i] += humanlike_oversteer_understeer_reward * (oversteer_understeer_score / 5.0) * _o_signal

            # Corner entry speed alignment — Brier-score penalty detected in pre-pass.
            # Fires only at corner entry steps; near-zero when agent matches target ratio.
            if humanlike_corner_entry_speed_reward != 0:
                reward_into[i] += corner_entry_speed_reward_at[i]

    for i in range(n_frames - 1):  # Loop over all frames that were generated
        # Switch memory buffer sometimes
        if random.random() < 0.1:
            list_to_fill = Experiences_For_Buffer_Test if random.random() < config_copy.buffer_test_ratio else Experiences_For_Buffer

        n_steps = min(n_steps_max, n_frames - 1 - i)
        if discard_non_greedy_actions_in_nsteps:
            try:
                first_non_greedy = rollout_results["action_was_greedy"][i + 1 : i + n_steps].index(False) + 1
                n_steps = min(n_steps, first_non_greedy)
            except ValueError:
                pass

        rewards = np.empty(n_steps_max).astype(np.float32)
        for j in range(n_steps):
            rewards[j] = (gamma**j) * reward_into[i + j + 1] + (rewards[j - 1] if j >= 1 else 0)

        state_img = rollout_results["frames"][i]
        state_float = rollout_results["state_float"][i]
        state_potential = get_potential(rollout_results["state_float"][i])

        # Get action that was played
        action = rollout_results["actions"][i]
        terminal_actions = float((n_frames - 1) - i) if "race_time" in rollout_results else math.inf
        next_state_has_passed_finish = ((i + n_steps) == (n_frames - 1)) and ("race_time" in rollout_results)

        if not next_state_has_passed_finish:
            next_state_img = rollout_results["frames"][i + n_steps]
            next_state_float = rollout_results["state_float"][i + n_steps]
            next_state_potential = get_potential(rollout_results["state_float"][i + n_steps])
        else:
            # It doesn't matter what next_state_img and next_state_float contain, as the transition will be forced to be final
            next_state_img = state_img
            next_state_float = state_float
            next_state_potential = 0

        list_to_fill.append(
            Experience(
                state_img,
                state_float,
                state_potential,
                action,
                n_steps,
                rewards,
                next_state_img,
                next_state_float,
                next_state_potential,
                gammas,
                terminal_actions,
            )
        )
    number_memories_added_train += len(Experiences_For_Buffer)
    if len(Experiences_For_Buffer) > 1:
        buffer.extend(Experiences_For_Buffer)
    elif len(Experiences_For_Buffer) == 1:
        buffer.add(Experiences_For_Buffer[0])
    number_memories_added_test += len(Experiences_For_Buffer_Test)
    if len(Experiences_For_Buffer_Test) > 1:
        buffer_test.extend(Experiences_For_Buffer_Test)
    elif len(Experiences_For_Buffer_Test) == 1:
        buffer_test.add(Experiences_For_Buffer_Test[0])

    return buffer, buffer_test, number_memories_added_train, number_memories_added_test
