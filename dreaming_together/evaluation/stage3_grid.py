"""Stage 3 — the measured evaluation grid.

W(condition, dt_coord) over the certified stacks, against the frozen
calibrated opponent (identical for every cell), with per-episode logging
sufficient for P1-P4, P6, P7 and the report's bootstrap CIs.

Documented scope decisions (pre-committed ladder + deviations):
  - Fixed-opponent evaluation replaces co-evolving self-play: every
    condition faces the same frozen EliteScriptedTeam, which is
    statistically cleaner for the W comparison and removes co-evolution
    as a confound. Deviation from the design's Stage-3 self-play is
    recorded in REPORT.md.
  - Stacks are trained at dt_coord = 250 ms and EVALUATED across the rate
    sweep without per-rate fine-tuning (ladder: fine-tunes cut).
  - 500 episodes per cell (ladder rung 3), full 6-rate sweep.
  - z_g-zeroed cells at 250 ms per condition for the causal reference.

Per-episode record: condition, rate_ms, zeroed, seed, win, steps,
n_messages, wq_open_frac (fraction of coordinator periods with the
window open — the P7 timing proxy).

Run: python -m dreaming_together.evaluation.stage3_grid [--smoke]
Output: results/grid.csv (incremental, crash-safe append)
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RATES_MS = (50, 100, 250, 500, 1000, 2000)
DT_POLICY_MS = 50
EPISODES = 500
CSV_PATH = ROOT / "results" / "grid.csv"

STACKS = {
    "A": ROOT / "runs" / "stage2_A",
    "B": ROOT / "runs" / "stage2_B_diff_s1",
    "C": ROOT / "runs" / "stage2_C_diff_s1",
}

_W = {}


def _winit():
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import EliteScriptedTeam
    from dreaming_together.policies.ff_policy import GaussianPolicy
    from dreaming_together.training.stage2_coordination import (
        LOBS, ACT_DIM, make_coordinator)
    from dreaming_together.training.stage2_diffusion import (
        make_listener, Scorer)
    _W["env"] = CombatEnv(seed=0, privileged_obs=True)
    _W["blue"] = EliteScriptedTeam(1)
    for cond, run in STACKS.items():
        if cond == "A":
            r0 = GaussianPolicy(LOBS, ACT_DIM, hidden=(64, 64))
            r1 = GaussianPolicy(LOBS, ACT_DIM, hidden=(64, 64))
            listeners = ("ff", r0, r1, None, None)
        else:
            r0, r1 = make_listener(), make_listener()
            q0, q1 = Scorer(), Scorer()
            q0.load_state_dict(torch.load(run / "q0_final.pt",
                                          weights_only=True))
            q1.load_state_dict(torch.load(run / "q1_final.pt",
                                          weights_only=True))
            q0.eval(); q1.eval()
            listeners = ("sas", r0, r1, q0, q1)
        r0.load_state_dict(torch.load(run / "r0_final.pt",
                                      weights_only=True))
        r1.load_state_dict(torch.load(run / "r1_final.pt",
                                      weights_only=True))
        r0.eval(); r1.eval()
        coord = make_coordinator(cond)
        coord.load_state_dict(torch.load(run / "coord_final.pt",
                                         weights_only=True))
        coord.eval()
        _W[cond] = (listeners, coord)


def _eval_roll(job):
    """job = (cond, rate_steps, zeroed, seed) → episode record dict."""
    from dreaming_together.training.stage2_coordination import (
        coord_state, Z_DIM)
    from dreaming_together.training.stage2_diffusion import sas_act
    from dreaming_together.envs.combat_env import WINDOW_OPEN_WQ
    cond, rate_steps, zeroed, seed = job
    env, blue = _W["env"], _W["blue"]
    (kind, r0, r1, q0, q1), coord = _W[cond]

    env.reset(seed=seed)
    z = np.zeros(Z_DIM, dtype=np.float32)
    step = n_msg = 0
    windows_open = windows_total = 0
    with torch.no_grad():
        while not env.done:
            if step % rate_steps == 0:
                windows_total += 1
                if env._team_wq(0) > WINDOW_OPEN_WQ:
                    windows_open += 1
                if not zeroed:
                    zt, *_ = coord(torch.from_numpy(
                        coord_state(env)).unsqueeze(0), sample=False)
                    z = zt[0].numpy()
                    n_msg += 1
            actions = blue.act(env)
            for p, pol, q in (("red0", r0, q0), ("red1", r1, q1)):
                o = np.concatenate([env.obs(p), z]).astype(np.float32)
                if kind == "ff":
                    actions[p] = torch.tanh(
                        pol.mean_net(torch.from_numpy(o))).numpy()
                else:
                    a, _ = sas_act(pol, q, o, det=True)
                    actions[p] = a
            env.step(actions)
            step += 1
    return {
        "condition": cond,
        "rate_ms": rate_steps * DT_POLICY_MS,
        "zeroed": int(zeroed),
        "seed": seed,
        "win": int(tuple(env.team_result) == (1, -1)),
        "draw": int(tuple(env.team_result) == (0, 0)),
        "steps": step,
        "n_messages": n_msg,
        "window_open_frac": windows_open / max(1, windows_total),
    }


def done_cells(path: Path) -> dict:
    counts = {}
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["condition"], int(row["rate_ms"]),
                       int(row["zeroed"]))
                counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    args = ap.parse_args()
    n_eps = 3 if args.smoke else args.episodes
    rates = (100, 500) if args.smoke else RATES_MS

    cells = []
    for cond in ("A", "C", "B"):          # fastest first
        for rate in rates:
            cells.append((cond, rate, False))
        cells.append((cond, 250, True))    # causal reference
    CSV_PATH.parent.mkdir(exist_ok=True)
    have = done_cells(CSV_PATH)

    fields = ["condition", "rate_ms", "zeroed", "seed", "win", "draw",
              "steps", "n_messages", "window_open_frac"]
    write_header = not CSV_PATH.exists()
    fout = open(CSV_PATH, "a", newline="", buffering=1)
    writer = csv.DictWriter(fout, fieldnames=fields)
    if write_header:
        writer.writeheader()

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_winit)
    t0 = time.time()
    try:
        for cond, rate, zeroed in cells:
            key = (cond, rate, int(zeroed))
            already = have.get(key, 0)
            todo = n_eps - already
            if todo <= 0:
                print(f"cell {key}: complete ({already}), skipping",
                      flush=True)
                continue
            rate_steps = rate // DT_POLICY_MS
            jobs = [(cond, rate_steps, zeroed, 1_000_000 + rate * 1000
                     + int(zeroed) * 500_000 + k)
                    for k in range(already, n_eps)]
            tc = time.time()
            for rec in pool.imap_unordered(_eval_roll, jobs,
                                           chunksize=4):
                writer.writerow(rec)
            rows = todo
            wins = None
            print(f"cell {key}: +{rows} eps in {time.time()-tc:.0f}s "
                  f"(total elapsed {(time.time()-t0)/60:.0f} min)",
                  flush=True)
    finally:
        pool.close(); pool.join(); fout.close()
    print("GRID COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
