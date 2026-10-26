"""
The learner of the SAC pipeline, run in the main process of scripts/train_sac.py: takes scored rollouts from the workers, trains, shares the
new actor with the workers, writes TensorBoard curves, a CSV of every lap and the inputs of the laps worth keeping.
"""

import csv
import importlib
import time
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config_files import config_copy
from trackmania_rl import run_to_video
from trackmania_rl.action_adapter import input_timelines
from trackmania_rl.multiprocess import sac_common
from trackmania_rl.sac_learner import SACLearner, resolve_device

LAP_COLUMNS = [
    "time", "lap", "steps_played", "map", "explo", "fill_buffer", "finished", "lap_time_ms", "style_total", "failed", "fail_reason",
    "progress_m", "n_decisions", "reference", "lambda_s", "lambda_f", "new_fastest", "new_target",
]  # fmt: skip


def save_lap(directory: Path, name: str, rollout_results: dict, spec) -> None:
    """The lap as a TMInterface .inputs file (replay it in the game) and as arrays (actions, positions)."""
    directory.mkdir(parents=True, exist_ok=True)
    executed = np.asarray(rollout_results["executed_actions"], dtype=np.float64)
    brake, accel, steer = input_timelines(executed, spec)
    run_to_video.write_timelines_in_tmi_format(brake, accel, steer, directory / f"{name}.inputs")
    np.savez_compressed(
        directory / f"{name}.npz",
        executed_actions=executed,
        raw_actions=np.asarray(rollout_results["actions"][: len(executed)], dtype=np.float32),
        positions=np.asarray(rollout_results["positions"], dtype=np.float32),
        speeds_kmh=np.asarray(rollout_results["speeds_kmh"], dtype=np.float32),
    )


def sac_learner_process_fn(rollout_queues, shared_actor, shared_actor_lock, shared_steps, base_dir: Path, save_dir: Path, tensorboard_base_dir: Path):
    package = sac_common.load_driver_package(config_copy, base_dir)
    spec = sac_common.action_spec(config_copy)
    mean, std = sac_common.float_normalisation(config_copy)
    device = resolve_device(config_copy.sac_device)
    learner = SACLearner(
        config_copy, mean, std, (config_copy.H_downsized, config_copy.W_downsized), package.style.best_lap_time_ms, device=device
    )
    print(f"Driver: {package.describe()}; target: {config_copy.sac_target_gain_ms} ms faster than {package.style.best_lap_time_ms} ms, style cost <= {config_copy.sac_style_budget}")
    print(f"SAC on {device}; state of {config_copy.sac_float_input_dim} floats; decisions every {spec.decision_ms} ms")
    if learner.load(save_dir):
        print(" =====================     Learner state loaded !     ============================")
    with shared_actor_lock:
        shared_actor.load_state_dict(learner.net.actor.state_dict())

    writer = SummaryWriter(log_dir=str(tensorboard_base_dir / config_copy.run_name))
    laps_csv = save_dir / "laps.csv"
    new_file = not laps_csv.exists()
    queue_check_order = list(range(len(rollout_queues)))
    readers = [q._reader for q in rollout_queues]
    time_last_save = time.perf_counter()
    updates_at_last_sync = learner.stats["updates"]

    def sync_actor():
        with shared_actor_lock:
            shared_actor.load_state_dict(learner.net.actor.state_dict())
        torch.save(learner.actor_state(), save_dir / "actor.torch")

    with open(laps_csv, "a", newline="") as csv_file:
        lap_writer = csv.DictWriter(csv_file, fieldnames=LAP_COLUMNS, extrasaction="ignore")
        if new_file:
            lap_writer.writeheader()
        while True:
            wait(readers)
            for idx in queue_check_order:
                if not rollout_queues[idx].empty():
                    rollout_results, end_race_stats, score, fill_buffer, is_explo, map_name, map_status, rollout_duration, loop_number = rollout_queues[idx].get()
                    queue_check_order.append(queue_check_order.pop(queue_check_order.index(idx)))
                    break
            importlib.reload(config_copy)

            record = learner.add_rollout(rollout_results, end_race_stats, score, fill_buffer, is_explo, map_name)
            shared_steps.value = learner.stats["steps_played"]
            lap_writer.writerow(dict(record, time=time.strftime("%Y-%m-%d %H:%M:%S"), steps_played=learner.stats["steps_played"]))
            csv_file.flush()

            step = learner.stats["steps_played"]
            if record["finished"]:
                writer.add_scalar(f"lap/{map_name}_time_ms", record["lap_time_ms"], step)
            writer.add_scalar("lap/style_total", record["style_total"] if record["style_total"] is not None else 0.0, step)
            writer.add_scalar("lap/failed", float(record["failed"]), step)
            writer.add_scalar("lap/progress_m", record["progress_m"], step)
            writer.add_scalar("constraint/lambda_s", learner.multipliers.lambda_s, step)
            writer.add_scalar("constraint/lambda_f", learner.multipliers.lambda_f, step)
            time_text = f"{record['lap_time_ms'] / 1000:.3f} s" if record["finished"] else "not finished"
            style_text = f"{record['style_total']:.2f}" if record["style_total"] is not None else "-"
            print(
                f"lap {record['lap']:5d} {map_name} {'explo' if is_explo else 'eval '} {time_text:>13s}  style {style_text:>6s}  progress {record['progress_m']:6.0f} m"
                f"{'  FAILED: ' + record['fail_reason'] if record['failed'] else ''}  lambda_s {learner.multipliers.lambda_s:.3f} lambda_f {learner.multipliers.lambda_f:.3f}"
                f"  replay {len(learner.buffer)}"
            )

            if record["new_fastest"]:
                save_lap(save_dir / "fastest_laps", f"{map_name}_{record['lap_time_ms']}", rollout_results, spec)
            if record["new_target"]:
                print(f"  -> target lap: {record['lap_time_ms']} ms, style cost {record['style_total']:.2f}")
                save_lap(save_dir / "target_laps", f"{map_name}_{record['lap_time_ms']}_style{record['style_total']:.2f}", rollout_results, spec)

            due = learner.updates_due()
            if due > 0:
                t0 = time.perf_counter()
                chunk = max(1, config_copy.sac_actor_sync_every_updates)
                done = 0
                metrics = {}
                while done < due:
                    n = min(chunk, due - done)
                    metrics = learner.train(n)
                    done += n
                    if learner.stats["updates"] - updates_at_last_sync >= config_copy.sac_actor_sync_every_updates:
                        sync_actor()
                        updates_at_last_sync = learner.stats["updates"]
                for k, v in metrics.items():
                    writer.add_scalar(f"sac/{k}", v, learner.stats["updates"])
                print(f"  {done} updates in {time.perf_counter() - t0:.1f} s: " + ", ".join(f"{k} {v:.3g}" for k, v in metrics.items() if k in ("critic_loss", "actor_loss", "alpha", "q_mean", "entropy")))

            if time.perf_counter() - time_last_save > config_copy.sac_save_every_s:
                learner.save(save_dir)
                sync_actor()
                time_last_save = time.perf_counter()
