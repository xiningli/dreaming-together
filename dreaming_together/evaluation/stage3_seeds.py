"""Stage 3 addendum — multi-seed replication cells.

The design's 3-seed requirement: P1's ordering claim needs more than one
training run per condition. Seeds 2 and 3 (trained by the same certified
recipe: stage2_coordination --seed N for A, stage2_diffusion --seed N
for B/C with per-seed frozen channels) are evaluated here at the trained
rate (250 ms), live and z_g-zeroed, 500 episodes per cell, against the
identical frozen EliteScriptedTeam. Seed 1 is the original bring-up
stack, already measured in grid.csv — the report merges the two files.

Run: python -m dreaming_together.evaluation.stage3_seeds [--smoke]
Output: results/grid_seeds.csv (incremental, crash-safe append)
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

SEEDS = (2, 3)
RATE_MS = 250
DT_POLICY_MS = 50
EPISODES = 500
CSV_PATH = ROOT / "results" / "grid_seeds.csv"


def seed_run(cond: str, seed: int) -> Path:
    if cond == "A":
        return ROOT / "runs" / f"stage2_A_s{seed}"
    return ROOT / "runs" / f"stage2_{cond}_diff_s{seed}"


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
    for cond in ("A", "B", "C"):
        for seed in SEEDS:
            run = seed_run(cond, seed)
            if not (run / "coord_final.pt").exists():
                continue
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
            _W[(cond, seed)] = (listeners, coord)


def _eval_roll(job):
    from dreaming_together.training.stage2_coordination import (
        coord_state, Z_DIM)
    from dreaming_together.training.stage2_diffusion import sas_act
    from dreaming_together.envs.combat_env import WINDOW_OPEN_WQ
    cond, seed, zeroed, ep_seed = job
    env, blue = _W["env"], _W["blue"]
    (kind, r0, r1, q0, q1), coord = _W[(cond, seed)]
    rate_steps = RATE_MS // DT_POLICY_MS

    env.reset(seed=ep_seed)
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
        "seed": seed,
        "rate_ms": RATE_MS,
        "zeroed": int(zeroed),
        "ep_seed": ep_seed,
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
                key = (row["condition"], int(row["seed"]),
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

    cells = []
    for cond in ("A", "C", "B"):          # fastest first
        for seed in SEEDS:
            if not (seed_run(cond, seed) / "coord_final.pt").exists():
                print(f"stack {cond} s{seed} not trained yet, skipping",
                      flush=True)
                continue
            cells.append((cond, seed, False))
            cells.append((cond, seed, True))
    CSV_PATH.parent.mkdir(exist_ok=True)
    have = done_cells(CSV_PATH)

    fields = ["condition", "seed", "rate_ms", "zeroed", "ep_seed", "win",
              "draw", "steps", "n_messages", "window_open_frac"]
    write_header = not CSV_PATH.exists()
    fout = open(CSV_PATH, "a", newline="", buffering=1)
    writer = csv.DictWriter(fout, fieldnames=fields)
    if write_header:
        writer.writeheader()

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_winit)
    t0 = time.time()
    try:
        for cond, seed, zeroed in cells:
            key = (cond, seed, int(zeroed))
            already = have.get(key, 0)
            todo = n_eps - already
            if todo <= 0:
                print(f"cell {key}: complete ({already}), skipping",
                      flush=True)
                continue
            jobs = [(cond, seed, zeroed,
                     5_000_000 + seed * 100_000 + int(zeroed) * 50_000 + k)
                    for k in range(already, n_eps)]
            tc = time.time()
            for rec in pool.imap_unordered(_eval_roll, jobs, chunksize=4):
                writer.writerow(rec)
            print(f"cell {key}: +{todo} eps in {time.time()-tc:.0f}s "
                  f"(total elapsed {(time.time()-t0)/60:.0f} min)",
                  flush=True)
    finally:
        pool.close(); pool.join(); fout.close()
    print("SEED GRID COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
