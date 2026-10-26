"""Pieces shared by the SAC collector and learner processes."""

from pathlib import Path

import numpy as np

from trackmania_rl.action_adapter import ActionSpec
from trackmania_rl.driver_context import N_REFERENCE_FEATURES, DriverPackage, reference_feature_mean_std
from trackmania_rl.style_constraint import RewardWeights, StyleConstraint


def action_spec(cfg) -> ActionSpec:
    return ActionSpec(ticks_per_action=cfg.sac_ticks_per_action, steer_step=cfg.sac_steer_step)


def load_driver_package(cfg, base_dir: Path) -> DriverPackage:
    """The driver package named in the config (a path relative to the repository root, which is the parent of base_dir = linesight/)."""
    if not cfg.sac_driver_package:
        raise ValueError("set sac_driver_package in config.py: SAC measures the style of every lap against a driver (scripts/build_driver_package.py)")
    path = Path(cfg.sac_driver_package)
    if not path.is_absolute():
        path = base_dir.parent / path
    if cfg.sac_n_reference_features != N_REFERENCE_FEATURES:
        raise ValueError(f"sac_n_reference_features = {cfg.sac_n_reference_features} but the code produces {N_REFERENCE_FEATURES}")
    return DriverPackage.load(path)


def make_constraint(cfg, package: DriverPackage) -> StyleConstraint:
    weights = RewardWeights(w_progress=cfg.sac_w_progress, w_line=cfg.sac_w_line, w_time=cfg.sac_w_time)
    return StyleConstraint(package.style, action_spec(cfg), weights)


def float_normalisation(cfg):
    """Mean and std of the float state: the original ones plus the (already scaled) reference features."""
    mean, std = np.asarray(cfg.float_inputs_mean, dtype=np.float32), np.asarray(cfg.float_inputs_std, dtype=np.float32)
    if cfg.sac_use_reference_context:
        ref_mean, ref_std = reference_feature_mean_std()
        mean, std = np.concatenate([mean, ref_mean]), np.concatenate([std, ref_std])
    assert len(mean) == len(std) == cfg.sac_float_input_dim, (len(mean), cfg.sac_float_input_dim)
    return mean, std
