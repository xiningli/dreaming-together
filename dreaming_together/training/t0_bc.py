"""Trainability ladder T0 — behavior-clone the IK expert.

Pure supervised learning: (target point in hull frame) → (pan, tilt).
If this fails, the bug is in data or model plumbing, nothing else.

Gate: held-out muzzle-ray hit rate ≥ 0.95 when the cloned policy aims and
fires in the real physics env (the same AimTask used by T1).

Run: python -m dreaming_together.training.t0_bc
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.policies.ff_policy import MLP
from dreaming_together.training.aim_task import AimTask
from tools.ik_expert import aim_angles

RUN_DIR = Path(__file__).parent.parent.parent / "runs" / "t0"


def make_dataset(task: AimTask, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample n episodes; label each obs with the IK expert's action."""
    X = np.zeros((n, 3), dtype=np.float32)
    Y = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        X[i] = task.reset()
        pan, tilt = aim_angles(task.model, task.data, "red1", task.target)
        Y[i] = task.angles_to_act(pan, tilt)
    return X, Y


def main(n_train: int = 8000, n_epochs: int = 200, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    task = AimTask(seed=seed)
    X, Y = make_dataset(task, n_train)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)

    policy = MLP(3, 2)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = torch.mean((torch.tanh(policy(Xt)) - Yt) ** 2)
        loss.backward()
        opt.step()
    final_mse = float(loss.item())

    # gate: fire in the real env on held-out targets
    eval_task = AimTask(seed=seed + 1)
    hits = 0
    n_eval = 200
    with torch.no_grad():
        for _ in range(n_eval):
            obs = eval_task.reset()
            a = torch.tanh(policy(torch.from_numpy(obs))).numpy()
            _, hit = eval_task.fire(a)
            hits += int(hit)
    hit_rate = hits / n_eval

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), RUN_DIR / "policy.pt")
    result = {"final_mse": final_mse, "hit_rate": hit_rate, "n_eval": n_eval}
    print(f"T0 BC: train MSE={final_mse:.5f}, "
          f"held-out physical hit rate={hit_rate:.1%} ({hits}/{n_eval})")
    gate = hit_rate >= 0.95
    print(f"T0 gate ({'PASS' if gate else 'FAIL'}): hit rate "
          f"{'≥' if gate else '<'} 0.95")
    return result


if __name__ == "__main__":
    main()
