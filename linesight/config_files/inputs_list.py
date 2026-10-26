"""
The actions the agent can choose from: a pedal combination and a steering level.

Steering is given as a fraction of full lock in [-1, 1] (negative = left). With the default levels (-1, 0, 1) the list is the original
12 keyboard-style actions in the original order. More levels, e.g. (-1, -0.5, 0, 0.5, 1), give 20 actions and are sent to the game as analog
steering (see `analog_steering` in config.py and the CSetInputStateAnalog message in Python_Link.as).

Every action dict also keeps the boolean "left" / "right" keys (True for any steering to that side), so code that only
needs the direction of the steering keeps working.
"""

# (name, accelerate, brake), in the original order of the list
PEDAL_COMBOS = [
    ("Forward", True, False),
    ("Nothing", False, False),
    ("Brake", False, True),
    ("Brake and accelerate", True, True),
]


def make_inputs(steer_levels=(-1.0, 0.0, 1.0)):
    """
    Returns (inputs, action_forward_idx, action_backward_idx, n_forward_actions).

    Within each pedal combination the steering levels are ordered: straight, full left, full right, then the intermediate levels by decreasing
    size (left before right). n_forward_actions is the number of actions of the first combination (accelerate only), i.e. the
    actions with index < n_forward_actions are the "forward" ones.
    """
    levels = sorted({float(s) for s in steer_levels})
    if 0.0 not in levels or not all(-1.0 <= s <= 1.0 for s in levels):
        raise ValueError("steer_levels must contain 0 and lie in [-1, 1]")
    order = [0.0] + sorted((s for s in levels if s != 0.0), key=lambda s: (-abs(s), s))
    inputs = []
    for _, accelerate, brake in PEDAL_COMBOS:
        for steer in order:
            inputs.append(
                {
                    "left": steer < 0,
                    "right": steer > 0,
                    "accelerate": accelerate,
                    "brake": brake,
                    "steer": steer,
                }
            )
    forward = next(i for i, a in enumerate(inputs) if a["accelerate"] and not a["brake"] and a["steer"] == 0)
    backward = next(i for i, a in enumerate(inputs) if a["brake"] and not a["accelerate"] and a["steer"] == 0)
    return inputs, forward, backward, len(order)


inputs, action_forward_idx, action_backward_idx, n_forward_actions = make_inputs()
