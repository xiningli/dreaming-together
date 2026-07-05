"""Trainability ladder T1 — RL from scratch on the aim task.

REINFORCE with per-batch advantage normalization, Gaussian policy, no BC
warm start: this rung exists to prove that reward computed from real
physics (ray-swept pellet hits) produces a gradient that reaches the
policy and improves it (rules R2, R4). If T1's curve is flat, the
environment/reward plumbing is broken; fix it before anything larger runs.

Lessons already encoded here (each one stalled an earlier T1 run):
  - Shaping must be unsaturated (linear −miss), or random-init episodes
    get zero gradient.
  - Advantages are normalized per batch; a global EMA baseline mixes the
    randomized target difficulty into the learning signal.
  - The gate evaluates the DETERMINISTIC policy (mean action). Evaluating
    stochastic samples caps the measurable hit rate at the exploration
    noise level regardless of how good the mean is.

Kill criteria (rule R9), checked automatically:
  - NaN/inf in loss → abort
  - training hit rate still 0 after 25% of the iteration budget → abort

Gate: deterministic hit rate ≥ 0.90 over 200 held-out episodes.

Run: python -m dreaming_together.training.t1_aim_rl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.policies.ff_policy import GaussianPolicy
from dreaming_together.training.aim_task import AimTask

RUN_DIR = Path(__file__).parent.parent.parent / "runs" / "t1"


def evaluate(policy: GaussianPolicy, seed: int, n_eval: int = 200) -> float:
    """Deterministic (mean-action) hit rate on held-out targets."""
    task = AimTask(seed=seed)
    hits = 0
    with torch.no_grad():
        for _ in range(n_eval):
            obs = task.reset()
            mean = torch.tanh(policy.mean_net(torch.from_numpy(obs)))
            _, hit = task.fire(mean.numpy())
            hits += int(hit)
    return hits / n_eval


def main(n_iters: int = 600, batch: int = 64, seed: int = 0,
         lr: float = 3e-3) -> dict:
    torch.manual_seed(seed)
    task = AimTask(seed=seed)
    policy = GaussianPolicy(obs_dim=3, act_dim=2, init_log_std=-1.0)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    history: list[dict] = []
    t_start = time.time()

    for it in range(n_iters):
        obs_buf = np.zeros((batch, 3), dtype=np.float32)
        act_buf = np.zeros((batch, 2), dtype=np.float32)
        rew_buf = np.zeros(batch, dtype=np.float32)
        hit_buf = np.zeros(batch, dtype=bool)

        with torch.no_grad():
            for b in range(batch):
                obs = task.reset()
                a, _ = policy.act(torch.from_numpy(obs))
                r, hit = task.fire(a.numpy())
                obs_buf[b], act_buf[b] = obs, a.numpy()
                rew_buf[b], hit_buf[b] = r, hit

        rew_t = torch.from_numpy(rew_buf)
        adv = (rew_t - rew_t.mean()) / (rew_t.std() + 1e-6)

        d = policy.dist(torch.from_numpy(obs_buf))
        logp = d.log_prob(torch.from_numpy(act_buf)).sum(-1)
        loss = -(logp * adv).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(f"T1 kill criterion: non-finite loss at iter {it}")

        opt.zero_grad()
        loss.backward()
        opt.step()

        hit_rate = float(hit_buf.mean())
        history.append({"iter": it, "hit_rate": hit_rate,
                        "mean_reward": float(rew_buf.mean())})
        if it == n_iters // 4:
            recent = np.mean([h["hit_rate"] for h in history[-20:]])
            if recent == 0.0:
                raise RuntimeError(
                    "T1 kill criterion: hit rate flat at 0 after 25% of "
                    "budget — reward plumbing is broken, diagnose before "
                    "scaling up")
        if it % 50 == 0 or it == n_iters - 1:
            recent = np.mean([h["hit_rate"] for h in history[-20:]])
            print(f"iter {it:4d}  train_hit(last20)={recent:.2f}  "
                  f"reward={rew_buf.mean():.3f}  "
                  f"std={policy.log_std.exp().mean():.3f}")

    wall = time.time() - t_start
    det_hit_rate = evaluate(policy, seed=seed + 1)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), RUN_DIR / "policy.pt")
    (RUN_DIR / "history.json").write_text(json.dumps(history))

    gate = det_hit_rate >= 0.90
    print(f"\nT1 RL: deterministic hit rate={det_hit_rate:.1%} on 200 "
          f"held-out targets; {n_iters * batch} training episodes in "
          f"{wall:.0f}s ({n_iters * batch / wall:.0f} eps/s)")
    print(f"T1 gate ({'PASS' if gate else 'FAIL'}): deterministic hit rate "
          f"{'≥' if gate else '<'} 0.90")
    return {"det_hit_rate": det_hit_rate, "wall_s": wall, "history": history}


if __name__ == "__main__":
    main()
