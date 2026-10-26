# Analog steering levels for the IQN agent

The agent still picks one of a fixed list of actions (IQN needs that), but the steering part of each action can now be a fraction of
full lock instead of only left / straight / right.

## Why 5 levels

Quantising the real steering of the 187 analog (pad or wheel) Hockolicious laps, over 102,121 steps above 25 m/s:

| steering levels | RMS error vs the real value | actions |
|---|--:|--:|
| 3 (-1, 0, 1) | 0.131 | 12 |
| 5 (-1, -0.5, 0, 0.5, 1) | 0.069 | 20 |
| 7 | 0.048 | 28 |
| 9 | 0.037 | 36 |

Those drivers use full lock 64 % of the time, steer less than 5 % of lock 14 % of the time, and the remaining 22 % in between. Going from 3
to 5 levels halves the error; each step after that gains less and adds 8 actions. Pedals stay on / off: TMNF and TMUF have no analog
throttle or brake (TMInterface docs), and humans keep the throttle on 99.8 % of the lap.

## What changed

| file | change |
|---|---|
| `config_files/inputs_list.py` | `make_inputs(steer_levels)` builds the action list. The default reproduces the original 12 actions in the same order |
| `config_files/config.py` | `steer_levels` (default `[-1, 0, 1]`), `analog_steering` (true as soon as a level other than -1, 0, 1 is used) |
| `trackmania_rl/tmi_interaction/Python_Link.as` | new message `CSetInputStateAnalog` (id 24): accelerate, brake, steer integer; calls `SetInputState(InputType::Steer, value)` |
| `trackmania_rl/tmi_interaction/tminterface2.py` | `set_input_state(..., steer, analog)` sends it as `round(65536 * steer)` |
| `trackmania_rl/tmi_interaction/game_instance_manager.py` | passes `analog`; the previous-action inputs use the amount of left / right steering (0, 0.5 or 1) instead of two booleans |
| `trackmania_rl/buffer_management.py` | the "forward actions" test uses `n_forward_actions` instead of the number 2 |
| `trackmania_rl/run_to_video.py` | writes `steer <value>` lines to the `.inputs` file when steering is analog |

The network needs no change: the head size is `len(inputs)`. The human-likeness penalties only look at whether an action steers left or
right, which still holds, so they keep working. They were written for key taps and have not been retuned for half levels.

## To switch it on

In `config_files/config.py` set `steer_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]`. The first run with a new action count must start from
scratch: the saved weights in `models/` have a 12-action output layer.

## What is tested and what is not

`python tests/test_analog_inputs.py` (from `linesight/`) checks, without the game, that the default list equals the original 12 actions, the
5-level list, the level validation, and that the message ids and sizes agree between the plugin and the Python client.

**Not tested: in the game.** Nobody has run it in TMInterface yet. The plugin assumes `InputType::Steer` exists in the TMInterface
plugin API of the version in use; if it does not, the plugin will fail to compile and say so. The first thing to do in game is set a
fixed level, drive, and check that the car steers the expected amount (for example with the Q-value / input display, or by comparing
the effective steering with `scripts/tools/steering_from_replay.py`). Also not updated: the Q-value widget video in `run_to_video.py`,
which assumes 12 actions.
