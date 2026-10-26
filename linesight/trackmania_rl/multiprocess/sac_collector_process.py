"""
A worker that drives one Trackmania instance with the SAC actor and hands scored rollouts to the learner.
The counterpart of collector_process.py for continuous actions.
"""

import importlib
import time
from itertools import chain, count, cycle
from pathlib import Path

import numpy as np
import torch
from torch import multiprocessing as mp

from config_files import config_copy
from trackmania_rl.utilities import set_random_seed


def sac_collector_process_fn(
    rollout_queue,
    shared_actor,
    shared_actor_lock,
    game_spawning_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tmi_port: int,
    process_number: int,
):
    from trackmania_rl.agents.sac import SACInferer
    from trackmania_rl.driver_context import RolloutStyleContext
    from trackmania_rl.map_loader import analyze_map_cycle, load_next_map_zone_centers
    from trackmania_rl.multiprocess import sac_common
    from trackmania_rl.sac_learner import make_sac_net, resolve_device
    from trackmania_rl.sac_rollout import SACExplorationPolicy, score_rollout
    from trackmania_rl.tmi_interaction import game_instance_manager

    set_random_seed(process_number)
    spec = sac_common.action_spec(config_copy)
    package = sac_common.load_driver_package(config_copy, base_dir)
    constraint = sac_common.make_constraint(config_copy, package)
    style_context = RolloutStyleContext(package, spec.decision_ms / 1000.0, seed=process_number) if config_copy.sac_use_reference_context else None

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        running_speed=config_copy.running_speed,
        run_steps_per_action=spec.ticks_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=tmi_port,
    )

    device = resolve_device(config_copy.sac_device)
    mean, std = sac_common.float_normalisation(config_copy)
    actor = make_sac_net(config_copy, mean, std, (config_copy.H_downsized, config_copy.W_downsized)).actor
    inferer = SACInferer(actor, device)
    policy = SACExplorationPolicy(inferer, random_steps=config_copy.sac_random_steps, seed=process_number)
    try:
        actor.load_state_dict(torch.load(save_dir / "actor.torch", map_location=device, weights_only=False))
    except Exception as e:
        print("Worker could not load the actor, starting from scratch:", e)

    def update_network():
        with shared_actor_lock:
            actor.load_state_dict(shared_actor.state_dict())

    actor.eval()
    map_cycle_str = str(config_copy.map_cycle)
    set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
    map_cycle_iter = cycle(chain(*config_copy.map_cycle))
    zone_centers_filename = None

    for _ in range(3):  # warm up the network
        inferer.act(
            np.random.randint(0, 255, size=(1, config_copy.H_downsized, config_copy.W_downsized), dtype=np.uint8),
            np.random.rand(config_copy.sac_float_input_dim).astype(np.float32),
            explore=True,
        )

    time_since_last_queue_push = time.perf_counter()
    for loop_number in count(1):
        importlib.reload(config_copy)
        tmi.max_minirace_duration_ms = config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms

        if str(config_copy.map_cycle) != map_cycle_str:
            map_cycle_str = str(config_copy.map_cycle)
            set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
            map_cycle_iter = cycle(chain(*config_copy.map_cycle))

        next_map_tuple = next(map_cycle_iter)
        if next_map_tuple[2] != zone_centers_filename:
            zone_centers = load_next_map_zone_centers(next_map_tuple[2], base_dir)
        map_name, map_path, zone_centers_filename, is_explo, fill_buffer = next_map_tuple
        map_status = "trained" if map_name in set_maps_trained else "blind"
        if Path(zone_centers_filename).name != Path(package.style.centerline_file).name:
            print(f"Warning: the map's zone centers ({zone_centers_filename}) are not the driver package's centerline ({package.style.centerline_file})")

        policy.explore = bool(is_explo)
        update_network()

        rollout_start_time = time.perf_counter()
        rollout_results, end_race_stats = tmi.rollout(
            exploration_policy=policy,
            map_path=map_path,
            zone_centers=zone_centers,
            update_network=update_network,
            continuous_spec=spec,
            style_context=style_context,
        )
        rollout_duration = time.perf_counter() - rollout_start_time
        rollout_results["worker_time_in_rollout_percentage"] = rollout_duration / (time.perf_counter() - time_since_last_queue_push)
        time_since_last_queue_push = time.perf_counter()
        print("", flush=True)

        if not tmi.last_rollout_crashed:
            score = score_rollout(constraint, rollout_results, end_race_stats, style_context.reference if style_context else None)
            rollout_queue.put((rollout_results, end_race_stats, score, fill_buffer, is_explo, map_name, map_status, rollout_duration, loop_number))
