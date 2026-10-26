"""
Offline checks for trackmania_rl/agents/sac.py on CPU. Run from linesight/:  python tests/test_sac.py   (needs torch)

The learning test uses a toy task: the state is a lateral error and its rate (in the float features; the image is blank), the action
steers against it. It checks that the critic / actor / temperature updates run and that the policy learns to hold the error near 0.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackmania_rl.agents.sac import (  # noqa: E402
    ReplayBuffer,
    SACInferer,
    SACNet,
    SACTrainer,
    make_nstep_transitions,
)

HW = (64, 64)  # small images keep the test fast; the conv stack gives 32 * 4 * 4 = 512 features
FLOAT_DIM = 4


def make_net(seed=0):
    torch.manual_seed(seed)
    return SACNet(FLOAT_DIM, np.zeros(FLOAT_DIM), np.ones(FLOAT_DIM), float_hidden_dim=64, hidden_dim=128, img_hw=HW)


def test_log_prob_matches_the_change_of_variables_formula():
    net = make_net()
    torch.manual_seed(1)
    feats = torch.randn(8, net.encoder.out_dim)
    mean, log_std_raw = net.actor.actor_head(feats).chunk(2, dim=-1)
    a, logp = net.actor.policy_from_features(feats)
    # recompute from the action: u = atanh(a); log p(a) = log N(u) - sum log(1 - a^2)
    from trackmania_rl.agents import sac

    log_std = sac.LOG_STD_MIN + 0.5 * (sac.LOG_STD_MAX - sac.LOG_STD_MIN) * (torch.tanh(log_std_raw) + 1)
    u = torch.atanh(a.clamp(-0.999999, 0.999999))
    ref = (-0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1) - torch.log(1 - a**2 + 1e-12).sum(-1)
    assert torch.allclose(logp, ref, atol=1e-2, rtol=1e-3), (logp, ref)
    assert a.abs().max() < 1.0


def test_actor_loss_does_not_touch_the_encoder_and_critic_loss_does():
    net = make_net()
    trainer = SACTrainer(net)
    rng = np.random.default_rng(0)
    batch = dict(
        img=torch.randint(0, 255, (16, 1, *HW), dtype=torch.uint8),
        floats=torch.randn(16, FLOAT_DIM),
        action=torch.rand(16, 3) * 2 - 1,
        reward_n=torch.randn(16),
        discount_n=torch.full((16,), 0.99),
        next_img=torch.randint(0, 255, (16, 1, *HW), dtype=torch.uint8),
        next_floats=torch.randn(16, FLOAT_DIM),
    )
    before = {k: v.clone() for k, v in net.encoder.state_dict().items()}
    actor_before = {k: v.clone() for k, v in net.actor.actor_head.state_dict().items()}
    target_before = {k: v.clone() for k, v in trainer.target.encoder.state_dict().items()}
    metrics = trainer.update(batch)
    assert all(np.isfinite(v) for v in metrics.values()), metrics
    assert any(not torch.equal(before[k], v) for k, v in net.encoder.state_dict().items()), "critic loss should train the encoder"
    assert any(not torch.equal(actor_before[k], v) for k, v in net.actor.actor_head.state_dict().items()), "the actor should move"
    moved = [k for k, v in trainer.target.encoder.state_dict().items() if k.startswith("img_head") and not torch.equal(target_before[k], v)]
    assert moved, "the target encoder should follow the online encoder (soft update)"
    # one soft update moves the target by tau * (online - target)
    k = moved[0]
    expected = target_before[k] * (1 - trainer.tau) + net.encoder.state_dict()[k] * trainer.tau
    assert torch.allclose(trainer.target.encoder.state_dict()[k], expected, atol=1e-6)


def test_actor_loss_sends_no_gradient_into_the_encoder():
    net = make_net()
    img = torch.randint(0, 255, (8, 1, *HW), dtype=torch.uint8)
    floats = torch.randn(8, FLOAT_DIM)
    f = net.encoder(img, floats).detach()  # what SACTrainer.update gives the actor and the temperature
    a, logp = net.actor.policy_from_features(f)
    q1, q2 = net.q_values(f, a)
    (logp.mean() - torch.min(q1, q2).mean()).backward()
    assert all(p.grad is None for p in net.encoder.parameters())
    assert any(p.grad is not None for p in net.actor.actor_head.parameters())


def test_nstep_transitions_by_hand():
    rewards = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # arriving in states 1..4
    nxt, r_n, disc = make_nstep_transitions(rewards, n_steps=2, gamma=0.5, ended_in_finish=True)
    assert nxt.tolist() == [2, 3, 4, 4]
    assert np.allclose(r_n, [1 + 0.5 * 2, 2 + 0.5 * 3, 3 + 0.5 * 4, 4])
    assert np.allclose(disc, [0.25, 0.25, 0.0, 0.0])  # the last two reach the finish: no bootstrap
    _, _, disc2 = make_nstep_transitions(rewards, n_steps=2, gamma=0.5, ended_in_finish=False)
    assert np.allclose(disc2, [0.25, 0.25, 0.25, 0.5])  # cut off, not finished: bootstrap from the last state


def test_replay_buffer_wraps_and_samples():
    buf = ReplayBuffer(5, FLOAT_DIM, HW)
    for i in range(8):
        buf.add(np.full((1, *HW), i, np.uint8), np.full(FLOAT_DIM, i, np.float32), np.zeros(3), float(i), 0.9, np.full((1, *HW), i + 1, np.uint8), np.zeros(FLOAT_DIM))
    assert len(buf) == 5 and sorted(buf.reward_n.tolist()) == [3, 4, 5, 6, 7]
    b = buf.sample(4, "cpu", np.random.default_rng(0))
    assert b["img"].shape == (4, 1, *HW) and b["img"].dtype == torch.uint8 and b["action"].shape == (4, 3)


def test_sac_learns_to_hold_the_toy_error_near_zero():
    net = make_net(0)
    trainer = SACTrainer(net, lr_critic=1e-3, lr_actor=1e-3, lr_alpha=1e-3, tau=0.02)
    inferer = SACInferer(net.actor, "cpu")
    buf = ReplayBuffer(4000, FLOAT_DIM, HW)
    rng = np.random.default_rng(0)
    blank = np.zeros((1, *HW), np.uint8)
    gamma = 0.95

    def feats(x, v):
        return np.array([x, v, 0.0, 0.0], dtype=np.float32)

    def run_episode(policy, steps=40):
        x, v, total = rng.uniform(-1, 1), 0.0, 0.0
        for _ in range(steps):
            f = feats(x, v)
            a = policy(f)
            v = 0.8 * v + 0.3 * a[0] + rng.normal(0, 0.02)  # steering pushes the error
            x = float(np.clip(x + v, -3, 3))
            r = -(x**2) - 0.01 * a[0] ** 2
            total += r
            buf.add(blank, f, a, r, gamma, blank, feats(x, v))
        return total / steps

    random_score = np.mean([run_episode(lambda f: rng.uniform(-1, 1, 3).astype(np.float32)) for _ in range(20)])
    for _ in range(50):
        run_episode(lambda f: inferer.act(blank, f, explore=True))
        for _ in range(40):
            trainer.update(buf.sample(64, "cpu", rng))
    learned = np.mean([run_episode(lambda f: inferer.act(blank, f, explore=False)) for _ in range(20)])
    assert learned > 0.25 * random_score, (random_score, learned)  # reward is negative: much closer to 0 than random actions
    assert learned > -0.25, learned


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
