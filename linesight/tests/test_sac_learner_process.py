"""
Smoke test of the SAC learner process (multiprocess/sac_learner_process.py) with scored rollouts from the fake game of
test_rollout_continuous.py, put into a real multiprocessing queue. Checks that it writes laps.csv, saves the fastest lap as an .inputs file,
trains, shares new actor weights, saves its state and TensorBoard logs. Run from linesight/:  python tests/test_sac_learner_process.py (~1 min)
"""

import csv
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LINESIGHT = ROOT / "linesight"
sys.path.insert(0, str(LINESIGHT))
sys.path.insert(0, str(LINESIGHT / "tests"))

# small replay, early learning, frequent syncs and saves; the process reloads config_copy.py on every lap, so the overrides go in the file
OVERRIDES = """
run_name = "smoke"
sac_device = "cpu"
sac_replay_capacity = 6000
sac_start_learning_after = 300
sac_batch_size = 16
sac_updates_per_transition = 0.02
sac_actor_sync_every_updates = 5
sac_save_every_s = 1
sac_hidden_dim = 64
sac_float_hidden_dim = 32
"""
CONFIG_COPY = LINESIGHT / "config_files" / "config_copy.py"
existed = CONFIG_COPY.exists()
CONFIG_COPY.write_text((LINESIGHT / "config_files" / "config.py").read_text() + "\n" + OVERRIDES)

import faulthandler  # noqa: E402

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

torch.set_num_threads(1)  # as in train_sac.py: a forked worker must not inherit a multi-threaded OpenMP pool
faulthandler.dump_traceback_later(300, exit=True)  # a hang shows where instead of timing out silently

import test_rollout_continuous as fake  # noqa: E402
from config_files import config_copy  # noqa: E402
from trackmania_rl.action_adapter import ActionSpec  # noqa: E402
from trackmania_rl.driver_context import DriverPackage, RolloutStyleContext  # noqa: E402
from trackmania_rl.multiprocess import sac_common  # noqa: E402
from trackmania_rl.multiprocess.sac_learner_process import sac_learner_process_fn  # noqa: E402
from trackmania_rl.sac_learner import make_sac_net  # noqa: E402
from trackmania_rl.sac_rollout import score_rollout  # noqa: E402


def make_scored_rollout():
    package = DriverPackage.load(fake.PACKAGE)
    spec = ActionSpec(ticks_per_action=4)
    manager, _ = fake.make_manager(package, 4)
    zone_centers = fake.prepare_zone_centers(manager)
    context = RolloutStyleContext(package, spec.decision_ms / 1000.0)
    k = {"n": 0}

    def policy(frame, floats):
        raw = fake.scripted_raw_action(k["n"])
        k["n"] += 1
        return raw, True, 0.0, np.zeros(1, dtype=np.float32)

    results, stats = manager.rollout(policy, fake.MAP, zone_centers, lambda: None, continuous_spec=spec, style_context=context)
    score = score_rollout(sac_common.make_constraint(config_copy, package), results, stats, context.reference)
    return results, stats, score


def test_learner_process_consumes_rollouts_and_writes_its_outputs():
    results, stats, score = make_scored_rollout()
    mean, std = sac_common.float_normalisation(config_copy)
    shared_actor = make_sac_net(config_copy, mean, std, (config_copy.H_downsized, config_copy.W_downsized)).actor
    shared_actor.share_memory()
    before = {k: v.clone() for k, v in shared_actor.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmp:
        save_dir, tb_dir = Path(tmp) / "save", Path(tmp) / "tb"
        save_dir.mkdir()
        queue = mp.Queue(2)
        proc = mp.Process(
            target=sac_learner_process_fn,
            args=([queue], shared_actor, mp.Lock(), mp.Value("l", 0), LINESIGHT, save_dir, tb_dir),
            daemon=True,
        )
        proc.start()
        try:
            for i in range(3):
                queue.put((results, stats, score, True, i != 2, "hock", "trained", 1.0, i + 1))
                time.sleep(0.5)
            laps_csv = save_dir / "laps.csv"
            deadline = time.time() + 120
            while time.time() < deadline:
                if laps_csv.exists() and len(list(csv.DictReader(open(laps_csv)))) >= 3 and (save_dir / "sac.torch").exists():
                    break
                time.sleep(1)
            assert proc.is_alive(), "the learner process died"
        finally:
            proc.terminate()
            proc.join(10)

        rows = list(csv.DictReader(open(save_dir / "laps.csv")))
        assert [r["lap"] for r in rows][:3] == ["1", "2", "3"] and rows[0]["finished"] == "True" and int(rows[0]["lap_time_ms"]) == stats["race_time"]
        assert rows[0]["new_fastest"] == "True" and rows[1]["new_fastest"] == "False" and rows[0]["reference"] == results["reference_lap_id"]
        assert [r["explo"] for r in rows[:3]] == ["True", "True", "False"]
        inputs = list((save_dir / "fastest_laps").glob("*.inputs"))
        assert len(inputs) == 1 and "press up" in inputs[0].read_text() and " steer " in inputs[0].read_text()
        saved = np.load(next((save_dir / "fastest_laps").glob("*.npz")))
        assert saved["executed_actions"].shape == (len(results["executed_actions"]), 3) and saved["positions"].shape[1] == 3
        for name in ("sac.torch", "learner_state.json", "actor.torch"):
            assert (save_dir / name).exists(), name
        assert any(tb_dir.rglob("events.out.tfevents*"))
        after = shared_actor.state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before), "the learner should have shared a trained actor with the workers"
        actor_file = torch.load(save_dir / "actor.torch", weights_only=False)
        assert set(actor_file) == set(before)


if __name__ == "__main__":
    try:
        test_learner_process_consumes_rollouts_and_writes_its_outputs()
        print("ok test_learner_process_consumes_rollouts_and_writes_its_outputs")
    finally:
        if not existed and CONFIG_COPY.exists():
            CONFIG_COPY.unlink()
