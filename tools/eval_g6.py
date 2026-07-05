"""G6 evaluation (standalone, corrected entropy criterion).

Why this exists: the in-training G6 check measured CONDITIONAL message
entropy against 0.5·H_max. A well-seeded protocol is nearly deterministic
per state — low conditional entropy with high mutual information is what
good communication looks like — and even the perfect scripted protocol
fails that bar. The design's entropy criterion guards against protocol
COLLAPSE (one message for all states), so the correct measure is marginal
message diversity across states, referenced to the scripted protocol:

  diversity(learned) ≥ 0.5 × diversity(scripted on the same states)

where diversity = summed per-position empirical token entropy.

Other checks unchanged: win ≥ 0.75 vs scripted duo (deterministic eval);
causal necessity: zeroing z_g drops win rate ≥ 15 points.

Run: python tools/eval_g6.py [--episodes 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dreaming_together.policies.ff_policy import GaussianPolicy
from dreaming_together.coordinators.vocab import MSG_LEN, VOCAB_SIZE
from dreaming_together.training.stage2_coordination import (
    LOBS, ACT_DIM, STATE_DIM, DT_COORD_STEPS, Z_DIM, coord_state,
    make_coordinator,
)


def marginal_entropy(msgs: np.ndarray) -> float:
    """Summed per-position empirical symbol entropy (nats).

    Condition C: MSG_LEN token positions over the 32-token vocab.
    Conditions A/B: the 5 bottleneck dims, discretized to 64 bins of the
    quantizer's [-1,1] range (finer binning is noise at eval sample sizes).
    """
    h = 0.0
    if np.issubdtype(msgs.dtype, np.integer):
        for pos in range(msgs.shape[1]):
            counts = np.bincount(msgs[:, pos], minlength=VOCAB_SIZE)
            p = counts / counts.sum()
            p = p[p > 0]
            h += float(-(p * np.log(p)).sum())
        return h
    bins = np.clip(((msgs + 1) / 2 * 64).astype(int), 0, 63)
    for pos in range(msgs.shape[1]):
        counts = np.bincount(bins[:, pos], minlength=64)
        p = counts / counts.sum()
        p = p[p > 0]
        h += float(-(p * np.log(p)).sum())
    return h


def run_episodes(r0, r1, coord, n, mode, condition="C", seed0=900_000):
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import EliteScriptedTeam
    from dreaming_together.oracle.scripted_coordinator import (
        scripted_tokens, scripted_bottleneck)
    env = CombatEnv(seed=0, privileged_obs=True)
    blue = EliteScriptedTeam(1)   # frozen calibrated opponent
    wins = 0
    learned_msgs, scripted_msgs = [], []
    with torch.no_grad():
        for ep in range(n):
            env.reset(seed=seed0 + ep)
            z = np.zeros(Z_DIM, dtype=np.float32)
            step = 0
            while not env.done:
                if step % DT_COORD_STEPS == 0:
                    s = coord_state(env)
                    scripted_msgs.append(
                        scripted_tokens(env, 0)[0].numpy()
                        if condition == "C"
                        else scripted_bottleneck(env, 0)[0].numpy())
                    if mode == "on":
                        zt, toks, _, _ = coord(
                            torch.from_numpy(s).unsqueeze(0), sample=False)
                        z = zt[0].numpy()
                        learned_msgs.append(toks[0].numpy())
                    else:
                        z = np.zeros(Z_DIM, dtype=np.float32)
                actions = blue.act(env)
                for p, pol in (("red0", r0), ("red1", r1)):
                    o = torch.from_numpy(np.concatenate([env.obs(p), z]))
                    actions[p] = torch.tanh(pol.mean_net(o)).numpy()
                env.step(actions)
                step += 1
            wins += int(tuple(env.team_result) == (1, -1))
    return wins / n, np.array(learned_msgs), np.array(scripted_msgs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--condition", default="C", choices=("A", "B", "C"))
    args = ap.parse_args()
    RUN = ROOT / "runs" / ("stage2" if args.condition == "C"
                           else f"stage2_{args.condition}")

    r0 = GaussianPolicy(LOBS, ACT_DIM, hidden=(64, 64))
    r1 = GaussianPolicy(LOBS, ACT_DIM, hidden=(64, 64))
    coord = make_coordinator(args.condition)
    r0.load_state_dict(torch.load(RUN / "r0_final.pt", weights_only=True))
    r1.load_state_dict(torch.load(RUN / "r1_final.pt", weights_only=True))
    coord.load_state_dict(torch.load(RUN / "coord_final.pt",
                                     weights_only=True))
    for m in (r0, r1, coord):
        m.eval()

    win_on, learned, scripted = run_episodes(
        r0, r1, coord, args.episodes, "on", args.condition)
    win_off, _, _ = run_episodes(r0, r1, coord, args.episodes, "off",
                                 args.condition)

    h_learned = marginal_entropy(learned)
    h_scripted = marginal_entropy(scripted)
    div_ratio = h_learned / max(h_scripted, 1e-9)
    drop = (win_on - win_off) * 100

    checks = {
        "win_rate": (win_on, win_on >= 0.75),
        "diversity_ratio": (div_ratio, div_ratio >= 0.5),
        "causal_drop_pts": (drop, drop >= 15.0),
    }
    print(f"win(z_g on)  = {win_on:.3f}")
    print(f"win(z_g off) = {win_off:.3f}   drop = {drop:.1f} pts")
    print(f"marginal msg entropy: learned {h_learned:.2f} nats, "
          f"scripted {h_scripted:.2f} nats, ratio {div_ratio:.2f}")
    ok = all(c for _, c in checks.values())
    for k, (v, c) in checks.items():
        print(f"G6 {k}: {v:.2f} → {'OK' if c else 'FAIL'}")
    result = {k: {"value": float(v), "ok": bool(c)}
              for k, (v, c) in checks.items()}
    result["pass"] = bool(ok)
    (RUN / "G6_RESULT_v2.json").write_text(json.dumps(result, indent=1))
    print(f"\nG6 gate (corrected criterion): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
