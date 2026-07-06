"""NC — the no-communication baseline (trained deaf).

The eval-time z_g-zeroing ablation understates what a communication-free
team can do: those policies were trained expecting messages. NC trains
the same listener architecture (feedforward, Stage-1-initialized, same
budget and recipe as condition A's Phase A) with the channel zeroed from
the first update, so the baseline reflects a team that has adapted to
silence. The (condition_zeroed − NC) gap then separates "protocol
dependence" from "task needs communication".

Run: python -m dreaming_together.training.baseline_nc
Output: runs/baseline_NC/{r0,r1}_final.pt
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dreaming_together.policies.ff_policy import MLP
from dreaming_together.training.stage2_coordination import (
    LOBS, make_listener, _worker_init, _rollout, ppo_update)
from dreaming_together.training.stage1_combat import _serialize

RUN = ROOT / "runs" / "baseline_NC"


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    r0 = make_listener(ROOT / "runs/stage1/shield/policy_final.pt")
    r1 = make_listener(ROOT / "runs/stage1/shotgun/policy_final.pt")
    v0 = MLP(LOBS, 1, hidden=(64, 64))
    v1 = MLP(LOBS, 1, hidden=(64, 64))
    opt = torch.optim.Adam(
        list(r0.parameters()) + list(v0.parameters())
        + list(r1.parameters()) + list(v1.parameters()), lr=3e-4)

    ctx = mp.get_context("spawn")
    # coordinator init bytes are required by the worker but unused in
    # zeroed mode; condition "C" instantiates a throwaway LanguageCoordinator
    import io
    from dreaming_together.training.stage2_coordination import (
        make_coordinator)
    buf = io.BytesIO()
    torch.save(make_coordinator("C").state_dict(), buf)
    pool = ctx.Pool(8, initializer=_worker_init, initargs=(buf.getvalue(),))

    t0 = time.time()
    try:
        best = 0.0
        best_state = (r0.state_dict(), r1.state_dict())
        for it in range(120):
            jobs = [(_serialize(r0), _serialize(r1), None,
                     600_000 + it * 1000 + k, "zeroed")
                    for k in range(32)]
            eps = pool.map(_rollout, jobs)
            for pol, val, key in ((r0, v0, "red0"), (r1, v1, "red1")):
                ppo_update(pol, val, opt, eps, key)
            win = float(np.mean([e[2]["win"] for e in eps]))
            if win > best:
                import copy
                best = win
                best_state = copy.deepcopy((r0.state_dict(),
                                            r1.state_dict()))
            if it % 10 == 0 or it == 119:
                print(f"NC it {it:3d}/120 win {win:.2f} best {best:.2f} "
                      f"({(time.time()-t0)/60:.0f} min)", flush=True)
            if it >= 20 and win >= 0.85:
                print("NC early stop", flush=True)
                break
        r0.load_state_dict(best_state[0])
        r1.load_state_dict(best_state[1])
        torch.save(r0.state_dict(), RUN / "r0_final.pt")
        torch.save(r1.state_dict(), RUN / "r1_final.pt")
        print(f"NC baseline saved (best train win {best:.2f} vs standard "
              f"opponent)")
    finally:
        pool.close(); pool.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
