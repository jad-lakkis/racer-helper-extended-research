"""
Soft Actor-Critic (SAC) for Trackmania with continuous actions: steering, throttle, brake.

The network reads the same inputs as the IQN agent: a 1 x 120 x 160 grey image and the vector of float features
(speed, orientation, look-ahead centerline points, previous actions, ...). It outputs a Gaussian over three numbers that is squashed
with tanh to [-1, 1]; trackmania_rl.action_adapter turns them into game inputs.

Design (the usual choices for pixel-based SAC):
  * one encoder (conv head on the image + MLP on the floats), trained by the critic loss only;
  * the actor head and the temperature read the encoder output without sending gradients into it;
  * twin Q heads Q(features, action) and a slowly moving target copy of the encoder and the Q heads;
  * automatic temperature (entropy) tuning towards target_entropy = -action_dim;
  * n-step returns: a stored transition carries the discounted n-step reward and the discount to apply to the bootstrap value
    (0 when the race ended inside the n steps), so make_nstep_transitions() can build them from a whole rollout.

Nothing here depends on the game or on config; the pieces are tested on a toy task in tests/test_sac.py.
"""

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
ACTION_DIM = 3  # steering, throttle, brake, as outputs in [-1, 1]


def _init_orthogonal(m: nn.Module, gain: float = 1.0):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


def _mlp(dims, final_gain: float = 1.0) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.LeakyReLU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    mlp = nn.Sequential(*layers)
    gain = nn.init.calculate_gain("leaky_relu", 1e-2)
    for m in mlp[:-1]:
        _init_orthogonal(m, gain)
    _init_orthogonal(mlp[-1], final_gain)
    return mlp


class Encoder(nn.Module):
    """Image head (same convolutions as the IQN network) and float head, concatenated."""

    def __init__(self, float_dim: int, float_hidden_dim: int, float_mean: np.ndarray, float_std: np.ndarray, img_hw: Tuple[int, int] = (120, 160)):
        super().__init__()
        c = [1, 16, 32, 64, 32]
        self.img_head = nn.Sequential(
            nn.Conv2d(c[0], c[1], 4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(c[1], c[2], 4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(c[2], c[3], 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(c[3], c[4], 3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.float_head = nn.Sequential(nn.Linear(float_dim, float_hidden_dim), nn.LeakyReLU(), nn.Linear(float_hidden_dim, float_hidden_dim), nn.LeakyReLU())
        gain = nn.init.calculate_gain("leaky_relu", 1e-2)
        for m in list(self.img_head) + list(self.float_head):
            _init_orthogonal(m, gain)
        with torch.no_grad():
            self.conv_out_dim = int(self.img_head(torch.zeros(1, 1, *img_hw)).shape[1])
        self.out_dim = self.conv_out_dim + float_hidden_dim
        self.register_buffer("float_mean", torch.as_tensor(np.asarray(float_mean), dtype=torch.float32))
        self.register_buffer("float_std", torch.as_tensor(np.asarray(float_std), dtype=torch.float32))

    def forward(self, img: torch.Tensor, floats: torch.Tensor) -> torch.Tensor:
        if img.dtype == torch.uint8:
            img = (img.float() - 128.0) / 128.0
        floats = (floats - self.float_mean) / self.float_std
        return torch.cat((self.img_head(img), self.float_head(floats)), dim=1)


class ActorNet(nn.Module):
    """What a data-collecting worker needs: the encoder and the actor head."""

    def __init__(self, encoder: Encoder, hidden_dim: int, action_dim: int = ACTION_DIM):
        super().__init__()
        self.encoder = encoder
        self.actor_head = _mlp([encoder.out_dim, hidden_dim, hidden_dim, 2 * action_dim], final_gain=0.01)

    def policy_from_features(self, features: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action in (-1, 1), log-probability of that action)."""
        mean, log_std = self.actor_head(features).chunk(2, dim=-1)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1.0)
        std = log_std.exp()
        pre = mean if deterministic else mean + std * torch.randn_like(mean)
        action = torch.tanh(pre)
        # Gaussian log-density of `pre`, corrected for the tanh squashing
        log_prob = (-0.5 * ((pre - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob = log_prob - (2 * (math.log(2) - pre - F.softplus(-2 * pre))).sum(-1)
        return action, log_prob

    def forward(self, img: torch.Tensor, floats: torch.Tensor, deterministic: bool = False):
        return self.policy_from_features(self.encoder(img, floats), deterministic)


class SACNet(nn.Module):
    def __init__(
        self,
        float_dim: int,
        float_mean: np.ndarray,
        float_std: np.ndarray,
        float_hidden_dim: int = 256,
        hidden_dim: int = 512,
        action_dim: int = ACTION_DIM,
        img_hw: Tuple[int, int] = (120, 160),
        init_alpha: float = 0.1,
    ):
        super().__init__()
        encoder = Encoder(float_dim, float_hidden_dim, float_mean, float_std, img_hw)
        self.actor = ActorNet(encoder, hidden_dim, action_dim)
        self.q1 = _mlp([encoder.out_dim + action_dim, hidden_dim, hidden_dim, 1])
        self.q2 = _mlp([encoder.out_dim + action_dim, hidden_dim, hidden_dim, 1])
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), dtype=torch.float32))

    @property
    def encoder(self) -> Encoder:
        return self.actor.encoder

    def q_values(self, features: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((features, action), dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def actor_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}


class SACTrainer:
    def __init__(
        self,
        net: SACNet,
        lr_critic: float = 3e-4,
        lr_actor: float = 3e-4,
        lr_alpha: float = 3e-4,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        grad_clip: float = 10.0,
    ):
        self.net = net
        self.target = copy.deepcopy(net)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.tau = tau
        self.grad_clip = grad_clip
        self.target_entropy = -float(ACTION_DIM) if target_entropy is None else target_entropy
        critic_params = list(net.encoder.parameters()) + list(net.q1.parameters()) + list(net.q2.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=lr_critic)
        self.actor_opt = torch.optim.Adam(net.actor.actor_head.parameters(), lr=lr_actor)
        self.alpha_opt = torch.optim.Adam([net.log_alpha], lr=lr_alpha)
        self._critic_params = critic_params

    @property
    def alpha(self) -> torch.Tensor:
        return self.net.log_alpha.exp()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        batch: img (B,1,H,W) uint8, floats (B,D), action (B,3) in [-1,1], reward_n (B,), discount_n (B,), next_img, next_floats.
        reward_n is the discounted sum of the n rewards; discount_n is gamma^n, or 0 if the race ended within the n steps.
        """
        net, target = self.net, self.target
        alpha = self.alpha.detach()

        with torch.no_grad():
            next_a, next_logp = net.actor(batch["next_img"], batch["next_floats"])
            tq1, tq2 = target.q_values(target.encoder(batch["next_img"], batch["next_floats"]), next_a)
            y = batch["reward_n"] + batch["discount_n"] * (torch.min(tq1, tq2) - alpha * next_logp)

        features = net.encoder(batch["img"], batch["floats"])
        q1, q2 = net.q_values(features, batch["action"])
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = torch.nn.utils.clip_grad_norm_(self._critic_params, self.grad_clip)
        self.critic_opt.step()

        # the actor and the temperature see the features without gradients into the encoder
        f = features.detach()
        pi_a, logp = net.actor.policy_from_features(f)
        pq1, pq2 = net.q_values(f, pi_a)
        actor_loss = (alpha * logp - torch.min(pq1, pq2)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_norm = torch.nn.utils.clip_grad_norm_(net.actor.actor_head.parameters(), self.grad_clip)
        self.actor_opt.step()

        alpha_loss = -(net.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            tgt_params = list(target.encoder.parameters()) + list(target.q1.parameters()) + list(target.q2.parameters())
            for t, s in zip(tgt_params, self._critic_params):
                t.mul_(1 - self.tau).add_(s, alpha=self.tau)

        return dict(
            critic_loss=float(critic_loss),
            actor_loss=float(actor_loss),
            alpha=float(self.alpha),
            q_mean=float(torch.min(q1, q2).mean()),
            target_mean=float(y.mean()),
            entropy=float(-logp.mean()),
            critic_grad_norm=float(critic_norm),
            actor_grad_norm=float(actor_norm),
        )


class SACInferer:
    """Runs the actor on one state. Used by the data-collecting workers."""

    def __init__(self, actor: ActorNet, device: str):
        self.actor = actor.to(device)
        self.device = device

    @torch.no_grad()
    def act(self, img_uint8: np.ndarray, floats: np.ndarray, explore: bool) -> np.ndarray:
        """img_uint8: (1, H, W); floats: (D,). Returns an action in [-1, 1]^3; sampled when exploring, the mean otherwise."""
        img = torch.from_numpy(img_uint8).unsqueeze(0).to(self.device)
        fl = torch.from_numpy(np.asarray(floats, dtype=np.float32)).unsqueeze(0).to(self.device)
        action, _ = self.actor(img, fl, deterministic=not explore)
        return action[0].cpu().numpy().astype(np.float32)

    def load_actor_state(self, state_dict: Dict[str, torch.Tensor]):
        self.actor.load_state_dict(state_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Replay
# ─────────────────────────────────────────────────────────────────────────────


class ReplayBuffer:
    """Uniform replay of n-step transitions. Images are kept as uint8."""

    def __init__(self, capacity: int, float_dim: int, img_hw: Tuple[int, int] = (120, 160), action_dim: int = ACTION_DIM):
        self.capacity = capacity
        self.img = np.zeros((capacity, 1, *img_hw), dtype=np.uint8)
        self.next_img = np.zeros((capacity, 1, *img_hw), dtype=np.uint8)
        self.floats = np.zeros((capacity, float_dim), dtype=np.float32)
        self.next_floats = np.zeros((capacity, float_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_n = np.zeros(capacity, dtype=np.float32)
        self.discount_n = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return self.size

    def add(self, img, floats, action, reward_n, discount_n, next_img, next_floats):
        i = self.pos
        self.img[i], self.floats[i], self.action[i] = img, floats, action
        self.reward_n[i], self.discount_n[i] = reward_n, discount_n
        self.next_img[i], self.next_floats[i] = next_img, next_floats
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str, rng: Optional[np.random.Generator] = None) -> Dict[str, torch.Tensor]:
        idx = (rng or np.random).integers(0, self.size, size=batch_size)
        t = lambda a: torch.from_numpy(a[idx]).to(device, non_blocking=True)  # noqa: E731
        return dict(
            img=t(self.img), floats=t(self.floats), action=t(self.action), reward_n=t(self.reward_n),
            discount_n=t(self.discount_n), next_img=t(self.next_img), next_floats=t(self.next_floats),
        )  # fmt: skip


def make_nstep_transitions(rewards_into: np.ndarray, n_steps: int, gamma: float, ended_in_finish: bool):
    """
    n-step targets for one rollout of T states (indices 0 .. T-1) and T-1 actions.

    rewards_into[i] is the reward received on arriving in state i (rewards_into[0] is unused).
    Returns, for each state i = 0 .. T-2: next index, reward_n and discount_n.
      reward_n[i]   = sum_{j < m} gamma^j rewards_into[i + 1 + j], with m = min(n_steps, T-1-i)
      next index    = i + m
      discount_n[i] = gamma^m, except 0 when the race was finished (ended_in_finish) and the n steps reach the last state;
                      a rollout cut short without finishing still bootstraps from its last state.
    """
    T = len(rewards_into)
    nxt = np.zeros(T - 1, dtype=np.int64)
    r_n = np.zeros(T - 1, dtype=np.float64)
    disc = np.zeros(T - 1, dtype=np.float64)
    for i in range(T - 1):
        m = min(n_steps, T - 1 - i)
        r_n[i] = sum(gamma**j * rewards_into[i + 1 + j] for j in range(m))
        nxt[i] = i + m
        disc[i] = 0.0 if (ended_in_finish and i + m == T - 1) else gamma**m
    return nxt, r_n.astype(np.float32), disc.astype(np.float32)
