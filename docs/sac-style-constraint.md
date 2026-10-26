# SAC with a driver style constraint

How the continuous-action agent is trained to drive like a chosen driver, and what has and has not been verified. Entry point: `linesight/scripts/train_sac.py`; all settings are the `sac_*` block of `linesight/config_files/config.py`.

## The idea

Faster laps are not enough: the agent should be fast **and** drive like a given human. Per corner, a driver has a style vector `z = (b, f, l, s)`: braking fraction, full-throttle fraction, line (entry, apex, exit, as a share of half the road width) and steering speed (see `style-vectors.md`). From the driver's five fastest laps we get a range for every metric in every corner. The problem is posed as a constrained MDP: maximise progress, subject to the style cost staying within a budget and the failure rate staying low.

```
r_t        = w_progress * (metres advanced) - w_line * |distance to the reference line| - w_time * dt
c_style    = at the exit of each corner: sum over b, f, l_entry, l_apex, l_exit, s of  distance outside [lo, hi] / max(hi - lo, floor)   (each term capped at 3)
c_fail     = 1 on the last step of a lap that did not finish, covers less than 90% of the reference line, or stays far from it
r_adjusted = r - lambda_s * c_style - lambda_f * c_fail                (all times sac_reward_scale)

after every sac_lambda_update_every_laps laps:
   lambda_s <- max(0, lambda_s + eta_s * (mean style cost per lap - B))
   lambda_f <- max(0, lambda_f + eta_f * (share of failed laps - eps))
```

A lap is an *output* (saved to `save/<run>/target_laps/`) when it is valid, at least `sac_target_gain_ms` faster than the driver's best, and its total style cost is within `B`. A valid lap that is merely the fastest so far goes to `fastest_laps/`.

The style of an agent lap is measured by **the same code** as for human replays (`trackmania_rl/style_features.py`), so the agent is held to numbers extracted the same way. The reward weights, floors and budgets are first guesses and will need tuning.

## Pieces

| File | Role |
|---|---|
| `trackmania_rl/action_adapter.py` | network output in [-1, 1]^3 -> (steer, throttle, brake); pedal levels k/4 -> duty pattern over the 4 ticks; input timelines on the 10 ms grid |
| `tmi_interaction/Python_Link.as`, `tminterface2.py` | message that sets analog steering and plays a 4-tick pedal pattern, one tick per physics step |
| `trackmania_rl/agents/sac.py` | encoder (same conv stack as IQN), tanh-Gaussian actor, twin Q, automatic temperature, n-step replay |
| `trackmania_rl/style_constraint.py` | `DriverStyle`, per-corner style cost, failure cost, reward, `Multipliers`, `select_target` |
| `trackmania_rl/driver_context.py`, `scripts/build_driver_package.py` | driver package (`style_vectors/<track>/packages/<driver>.npz`: ranges, corners, reference laps); 13 reference-line features in the state; line term of the reward |
| `GameInstanceManager.rollout(..., continuous_spec, style_context)` | continuous mode: sends the pattern, records positions / speeds / executed actions / finish position |
| `trackmania_rl/sac_rollout.py`, `sac_learner.py` | exploration policy with warm-up, scoring a rollout, transitions with the adjusted reward; replay, multipliers, target laps, save/load |
| `multiprocess/sac_collector_process.py`, `sac_learner_process.py` | one worker per game instance (scores each lap, queues it); the learner (trains, shares the actor, writes `laps.csv`, TensorBoard, `.inputs` files) |
| `scripts/tools/pedal_response_test.py` | in-game measurement of how the car answers pedal duty cycles |

## Running it

1. Extract style vectors (`scripts/extract_style_vectors.py`) and build packages: `python scripts/build_driver_package.py ../style_vectors/Hockolicious --top-drivers 5`. Packages for the fastest drivers of the four tracks are already committed.
2. Set `sac_driver_package` (and a `map_cycle` for the same track, with the centerline the package was built with) in `config.py`.
3. **Check the pedals first:** `python scripts/tools/pedal_response_test.py` on a map that starts with a straight. If the response is far from the level (e.g. a quarter duty cycle already brakes almost fully), change the pedal bins before training.
4. `python scripts/train_sac.py`. Per lap it prints the lap time, style cost, failure reason and multipliers; `save/<run>/laps.csv` has the same; `tensorboard --logdir tensorboard`.

State of the agent: the usual 184 floats of the IQN agent (previous actions are the executed ones) plus 13 reference features: the reference lap's line and speed at 0, 15, 30, 60, 100 and 150 m ahead, and the current distance to its line. One of the driver's reference laps is sampled for each rollout. Decisions are every 40 ms (`sac_ticks_per_action = 4`).

## What has been verified

Offline, all with the scripts under `linesight/tests` (run each as `python tests/<name>.py` from `linesight/`):

- `test_style_constraint.py`: a human replay played through the agent's 40 ms path gives the extractor's numbers (b, f and line within 0.02, steering speed within 0.3 on a 0-5.5 range); the driver's own lap costs 0.8 against 36 for flat-out driving without braking; multipliers, target selection.
- `test_driver_context.py`: packages load and round-trip; the step-by-step projection onto the centerline is bit-identical to the whole-lap one; the line term uses the reference line.
- `test_sac.py`, `test_sac_pipeline.py`: log-probability, gradients (the actor loss does not touch the encoder), n-step targets by hand, a toy control task that SAC solves; scoring and transitions of a rollout, the learner (replay, lambda updates, target and fastest laps, save/load).
- `test_rollout_continuous.py`: `GameInstanceManager.rollout` against a fake TMInterface that follows the centerline, in continuous mode (patterns sent, executed actions, positions, finish position, 197-float state, previous actions, scoring, transitions) and in the original discrete mode (state unchanged by the refactor).
- `test_sac_learner_process.py`: the learner process on a real queue (laps.csv, saved `.inputs`, actor shared back, state saved, TensorBoard).

## What has not

- **Nothing has run in the real game.** The fake game follows a fixed path; it cannot say whether SAC learns to drive, how long that takes or whether the hyperparameters are sensible.
- Whether the car answers pedal duty cycles like partial pedals (`pedal_response_test.py` measures this) and whether `InputType::Steer` in the plugin behaves as assumed.
- The reward scale, the style floors, `B`, `eps`, the multiplier step sizes and the replay size are untuned defaults. With laps of 1800 decisions and `sac_updates_per_transition = 0.5` a lap is followed by about 900 updates.
- The style ranges come from at most five laps of a driver (one lap on Interlagos, Catalunya and Monza GP), so on those tracks a range is a single point and the width floors do all the work. The Hockolicious drivers have 2 to 5 laps.
- The multipliers act on transitions as they enter the replay; older transitions keep the reward they were stored with.
- Speeds in the rollout are the norm of the car's velocity, not the game's display speed; they are used only to bound the projection step.
- Only one map per run is meant to work (the package belongs to one track). A warning is printed if the map's zone centers and the package centerline differ.
