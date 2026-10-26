"""
The learning side of the SAC pipeline without any multiprocessing: it takes scored rollouts, keeps the replay buffer and the Lagrange
multipliers, runs gradient updates and decides which laps are worth saving. The process in multiprocess/sac_learner_process.py
only moves rollouts from the queues into add_rollout() and calls train().

`cfg` is the config module (or anything with the same sac_* attributes).
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from trackmania_rl.agents.sac import ReplayBuffer, SACNet, SACTrainer
from trackmania_rl.sac_rollout import n_decisions, rollout_to_transitions
from trackmania_rl.style_constraint import LapScore, Multipliers, select_target


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def make_sac_net(cfg, float_mean: np.ndarray, float_std: np.ndarray, img_hw) -> SACNet:
    return SACNet(
        cfg.sac_float_input_dim,
        float_mean,
        float_std,
        float_hidden_dim=cfg.sac_float_hidden_dim,
        hidden_dim=cfg.sac_hidden_dim,
        img_hw=tuple(img_hw),
        init_alpha=cfg.sac_init_alpha,
    )


class SACLearner:
    def __init__(self, cfg, float_mean: np.ndarray, float_std: np.ndarray, img_hw, best_time_ms: Optional[int], device: str = "cpu", seed: int = 0):
        self.cfg = cfg
        self.device = device
        self.best_time_ms = best_time_ms  # the driver's best lap, the reference for the target gain
        self.net = make_sac_net(cfg, float_mean, float_std, img_hw).to(device)
        self.trainer = SACTrainer(
            self.net,
            lr_critic=cfg.sac_lr_critic,
            lr_actor=cfg.sac_lr_actor,
            lr_alpha=cfg.sac_lr_alpha,
            tau=cfg.sac_tau,
            target_entropy=cfg.sac_target_entropy,
        )
        self.buffer = ReplayBuffer(cfg.sac_replay_capacity, cfg.sac_float_input_dim, tuple(img_hw))
        self.multipliers = Multipliers(
            lambda_s=cfg.sac_lambda_s_init,
            lambda_f=cfg.sac_lambda_f_init,
            eta_s=cfg.sac_eta_s,
            eta_f=cfg.sac_eta_f,
            budget_style=cfg.sac_style_budget,
            budget_fail=cfg.sac_failure_budget,
        )
        self.rng = np.random.default_rng(seed)
        self.pending_scores: List[LapScore] = []
        self.last_multiplier_update: Optional[dict] = None
        self.laps: List[dict] = []
        self.stats = dict(steps_played=0, transitions_added=0, updates=0, laps=0, fastest_ms=None, best_target_ms=None)  # kept across restarts
        self._session_transitions = 0  # the replay is not saved: the update-to-data ratio counts from the start of this session
        self._session_updates = 0

    # ─── data in ────────────────────────────────────────────────────────────

    def add_rollout(self, rollout_results: dict, end_race_stats: dict, score: Optional[LapScore], fill_buffer: bool, is_explo: bool, map_name: str) -> dict:
        """Take one scored rollout. Returns the record of the lap (also appended to self.laps); `new_fastest` / `new_target` flag laps to save."""
        n = n_decisions(rollout_results)
        self.stats["steps_played"] += n
        self.stats["laps"] += 1
        record = dict(
            lap=self.stats["laps"],
            map=map_name,
            explo=bool(is_explo),
            fill_buffer=bool(fill_buffer),
            n_decisions=n,
            reference=rollout_results.get("reference_lap_id"),
            lambda_s=self.multipliers.lambda_s,
            lambda_f=self.multipliers.lambda_f,
            finished=False,
            lap_time_ms=None,
            style_total=None,
            failed=True,
            fail_reason="too short",
            progress_m=0.0,
            new_fastest=False,
            new_target=False,
        )
        if score is None:
            self.laps.append(record)
            return record
        record.update(
            finished=score.lap_time_ms is not None,
            lap_time_ms=score.lap_time_ms,
            style_total=score.style_total,
            failed=score.failed,
            fail_reason=score.fail_reason,
            progress_m=score.progress_m,
        )
        if fill_buffer:
            transitions = rollout_to_transitions(
                rollout_results, score, self.multipliers.lambda_s, self.multipliers.lambda_f, self.cfg.sac_n_steps, self.cfg.sac_gamma, self.cfg.sac_reward_scale
            )
            if transitions is not None:
                self.buffer.add_many(**transitions)
                self.stats["transitions_added"] += len(transitions["reward_n"])
                self._session_transitions += len(transitions["reward_n"])
            self.pending_scores.append(score)
            if len(self.pending_scores) >= self.cfg.sac_lambda_update_every_laps:
                self.last_multiplier_update = self.multipliers.update(self.pending_scores)
                self.pending_scores = []
        valid = record["finished"] and not record["failed"]
        if valid and (self.stats["fastest_ms"] is None or record["lap_time_ms"] < self.stats["fastest_ms"]):
            self.stats["fastest_ms"] = record["lap_time_ms"]
            record["new_fastest"] = True
        if valid and self.best_time_ms is not None and self.is_target(record):
            if self.stats["best_target_ms"] is None or record["lap_time_ms"] < self.stats["best_target_ms"]:
                self.stats["best_target_ms"] = record["lap_time_ms"]
                record["new_target"] = True
        self.laps.append(record)
        return record

    def is_target(self, record: dict) -> bool:
        """The lap is at least sac_target_gain_ms faster than the driver's best and within the style budget."""
        return select_target([record], self.best_time_ms, self.cfg.sac_target_gain_ms, self.cfg.sac_style_budget) is not None

    def best_target(self) -> Optional[dict]:
        return select_target(self.laps, self.best_time_ms, self.cfg.sac_target_gain_ms, self.cfg.sac_style_budget) if self.best_time_ms else None

    # ─── gradient updates ───────────────────────────────────────────────────

    def updates_due(self) -> int:
        """Updates owed by the update-to-data ratio, once the replay holds sac_start_learning_after transitions."""
        if len(self.buffer) < max(self.cfg.sac_start_learning_after, self.cfg.sac_batch_size):
            return 0
        owed = self._session_transitions * self.cfg.sac_updates_per_transition - self._session_updates
        return max(0, int(owed))

    def train(self, n_updates: int) -> dict:
        """Run n_updates SAC updates; returns the mean of each metric."""
        sums = {}
        for _ in range(n_updates):
            metrics = self.trainer.update(self.buffer.sample(self.cfg.sac_batch_size, self.device, self.rng))
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v
        self.stats["updates"] += n_updates
        self._session_updates += n_updates
        return {k: v / max(n_updates, 1) for k, v in sums.items()}

    def actor_state(self) -> dict:
        return self.net.actor_state_dict()

    # ─── files ──────────────────────────────────────────────────────────────

    def save(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainer.state_dict(), save_dir / "sac.torch")
        state = dict(stats=self.stats, multipliers=vars(self.multipliers), pending=len(self.pending_scores))
        (save_dir / "learner_state.json").write_text(json.dumps(state, indent=1))

    def load(self, save_dir: Path) -> bool:
        save_dir = Path(save_dir)
        if not (save_dir / "sac.torch").exists():
            return False
        self.trainer.load_state_dict(torch.load(save_dir / "sac.torch", map_location=self.device, weights_only=False))
        state = json.loads((save_dir / "learner_state.json").read_text())
        self.stats.update(state["stats"])
        for k, v in state["multipliers"].items():
            setattr(self.multipliers, k, v)
        return True
