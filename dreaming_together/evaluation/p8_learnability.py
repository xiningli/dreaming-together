"""P8 — the protocol-learnability (transmission) probe.

The evolutionary claim: compositional protocols survive transmission
bottlenecks. Operationalized: for conditions B and C, clone FRESH
listeners by behavior cloning on K logged episodes of the certified stack
playing with its live coordinator, then measure how much team win rate
the clones recover with the ORIGINAL coordinator still speaking.
Prediction P8: recovery(C) > recovery(B), gap widening as K shrinks.

Fairness: identical K, architecture (diffusion listener + scorer is the
certified stack; the clone here is the diffusion net BC'd on logged
(obs, chosen-action-horizon) pairs with the scorer REUSED — the scorer is
part of the listener pair, and what transmission must carry is the
message-conditioned behavior, which lives in the cloned net's
conditioning on z_g)... simpler and cleaner: clone BOTH pieces' effect by
BC-ing a fresh FF policy (obs283 → action5) per role on the logged pairs
— one architecture for both conditions, zero inherited weights, pure
"acquire the protocol from examples".

Run: python -m dreaming_together.evaluation.p8_learnability
Output: results/p8_learnability.csv
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dreaming_together.evaluation.stage3_grid import (
    STACKS, _winit, _W)

K_VALUES = (10, 50)
N_EVAL = 150
OUT = ROOT / "results" / "p8_learnability.csv"


def collect_logged(cond: str, n_episodes: int):
    """Episodes of the certified stack with its live coordinator; logs
    (obs283 incl. z_g, executed action5) for both red agents."""
    from dreaming_together.training.stage2_coordination import (
        coord_state, Z_DIM)
    from dreaming_together.training.stage2_diffusion import sas_act
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import EliteScriptedTeam
    (kind, r0, r1, q0, q1), coord = _W[cond]
    env = CombatEnv(seed=0, privileged_obs=True)
    blue = EliteScriptedTeam(1)
    X = {p: [] for p in ("red0", "red1")}
    Y = {p: [] for p in ("red0", "red1")}
    with torch.no_grad():
        for ep in range(n_episodes):
            env.reset(seed=3_000_000 + ep)
            z = np.zeros(Z_DIM, dtype=np.float32)
            step = 0
            while not env.done:
                if step % 5 == 0:
                    zt, *_ = coord(torch.from_numpy(
                        coord_state(env)).unsqueeze(0), sample=False)
                    z = zt[0].numpy()
                actions = blue.act(env)
                for p, pol, q in (("red0", r0, q0), ("red1", r1, q1)):
                    o = np.concatenate([env.obs(p), z]).astype(np.float32)
                    a, _ = sas_act(pol, q, o, det=True)
                    actions[p] = a
                    X[p].append(o); Y[p].append(a)
                env.step(actions)
                step += 1
    return X, Y


def clone_and_eval(cond: str, K: int) -> float:
    """BC fresh FF listeners on K episodes; eval with original coord."""
    from dreaming_together.policies.ff_policy import GaussianPolicy
    from dreaming_together.training.stage2_coordination import (
        LOBS, ACT_DIM, coord_state, Z_DIM)
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import EliteScriptedTeam
    torch.manual_seed(K)
    X, Y = collect_logged(cond, K)
    clones = {}
    for p in ("red0", "red1"):
        net = GaussianPolicy(LOBS, ACT_DIM, hidden=(64, 64))
        Xt = torch.tensor(np.array(X[p]), dtype=torch.float32)
        Yt = torch.tensor(np.clip(np.array(Y[p]), -.999, .999),
                          dtype=torch.float32)
        opt = torch.optim.Adam(net.mean_net.parameters(), lr=1e-3)
        for _ in range(1500):
            mb = torch.randint(0, len(Xt), (min(256, len(Xt)),))
            loss = torch.mean((torch.tanh(net.mean_net(Xt[mb]))
                               - Yt[mb]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        clones[p] = net

    (_, _, _, _, _), coord = _W[cond]
    env = CombatEnv(seed=0, privileged_obs=True)
    blue = EliteScriptedTeam(1)
    wins = 0
    with torch.no_grad():
        for ep in range(N_EVAL):
            env.reset(seed=4_000_000 + ep)
            z = np.zeros(Z_DIM, dtype=np.float32)
            step = 0
            while not env.done:
                if step % 5 == 0:
                    zt, *_ = coord(torch.from_numpy(
                        coord_state(env)).unsqueeze(0), sample=False)
                    z = zt[0].numpy()
                actions = blue.act(env)
                for p in ("red0", "red1"):
                    o = np.concatenate([env.obs(p), z]).astype(np.float32)
                    actions[p] = torch.tanh(clones[p].mean_net(
                        torch.from_numpy(o))).numpy()
                env.step(actions)
                step += 1
            wins += int(tuple(env.team_result) == (1, -1))
    return wins / N_EVAL


def main() -> int:
    _winit()   # load stacks into this process
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "K", "clone_win_rate"])
        for cond in ("C", "B"):
            for K in K_VALUES:
                wr = clone_and_eval(cond, K)
                print(f"P8 {cond} K={K}: clone team win {wr:.3f}",
                      flush=True)
                w.writerow([cond, K, wr])
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
