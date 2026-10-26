"""
Train a continuous-action SAC agent that drives like a chosen driver (the style constraint of the problem formulation).
The counterpart of scripts/train.py; the settings are the sac_* block of config_files/config.py. Needs the game, TMInterface and a driver
package (scripts/build_driver_package.py). See docs/sac-style-constraint.md.

    python scripts/train_sac.py
"""

import shutil
from pathlib import Path


def copy_configuration_file():
    base_dir = Path(__file__).resolve().parents[1]
    shutil.copyfile(base_dir / "config_files" / "config.py", base_dir / "config_files" / "config_copy.py")


if __name__ == "__main__":
    copy_configuration_file()

import ctypes
import os
import signal
import sys

import torch
import torch.multiprocessing as mp
from art import tprint
from torch.multiprocessing import Lock

from config_files import config_copy
from trackmania_rl.multiprocess import sac_common
from trackmania_rl.multiprocess.sac_collector_process import sac_collector_process_fn
from trackmania_rl.multiprocess.sac_learner_process import sac_learner_process_fn
from trackmania_rl.sac_learner import make_sac_net
from trackmania_rl.utilities import set_random_seed

torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
set_random_seed(444)


def clear_tm_instances():
    os.system("pkill -9 TmForever.exe" if config_copy.is_linux else "taskkill /F /IM TmForever.exe")


def signal_handler(sig, frame):
    print("Received SIGINT signal. Killing all open Trackmania instances.")
    clear_tm_instances()
    for child in mp.active_children():
        child.kill()
    tprint("Bye bye!", font="tarty1")
    sys.exit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    clear_tm_instances()

    base_dir = Path(__file__).resolve().parents[1]
    save_dir = base_dir / "save" / config_copy.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_base_dir = base_dir / "tensorboard"

    # Fail early (before any game is started) if the driver package or the state size does not fit
    package = sac_common.load_driver_package(config_copy, base_dir)
    sac_common.float_normalisation(config_copy)
    print(package.describe())

    # The plugin that plays the pedal duty cycles and the analog steering
    config_copy.target_python_link_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_dir / "trackmania_rl" / "tmi_interaction" / "Python_Link.as", config_copy.target_python_link_path)

    print("Run:\n\n")
    tprint(config_copy.run_name, font="tarty4")
    print("\n" * 2)
    print("SAC training with a driver style constraint is starting!")

    if config_copy.is_linux:
        os.system(f"chmod +x {config_copy.linux_launch_game_path}")

    shared_steps = mp.Value(ctypes.c_int64)
    shared_steps.value = 0
    rollout_queues = [mp.Queue(config_copy.max_rollout_queue_size) for _ in range(config_copy.gpu_collectors_count)]
    shared_actor_lock = Lock()
    game_spawning_lock = Lock()
    mean, std = sac_common.float_normalisation(config_copy)
    shared_actor = make_sac_net(config_copy, mean, std, (config_copy.H_downsized, config_copy.W_downsized)).actor
    shared_actor.share_memory()

    collector_processes = [
        mp.Process(
            target=sac_collector_process_fn,
            args=(rollout_queue, shared_actor, shared_actor_lock, game_spawning_lock, shared_steps, base_dir, save_dir, config_copy.base_tmi_port + process_number, process_number),
        )
        for rollout_queue, process_number in zip(rollout_queues, range(config_copy.gpu_collectors_count))
    ]
    for collector_process in collector_processes:
        collector_process.start()

    sac_learner_process_fn(rollout_queues, shared_actor, shared_actor_lock, shared_steps, base_dir, save_dir, tensorboard_base_dir)

    for collector_process in collector_processes:
        collector_process.join()
