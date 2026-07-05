"""Gate G2 — combat-signal statistics (rule R2, kills failure F2).

Scripted oracle vs scripted oracle over N randomized-spawn episodes.
The v0 run trained for hours on an environment where combat was
geometrically impossible; this gate certifies, before any learning run,
that the environment produces combat signal across the spawn distribution:

  PASS requires (design §6 G2):
    - pellet hits > 0 in ≥ 90% of episodes
    - mean HP damage per episode > 30
    - non-draw rate ≥ 60%
    - each team wins ≥ 20% of episodes (mirror-symmetry sanity)

Run: python tools/gate_g2.py [--episodes 200] [--video]
--video also records the first episode to videos/g2_oracle_episode.mp4.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from dreaming_together.envs.combat_env import CombatEnv, PREFIXES
from dreaming_together.oracle import ScriptedTeam


def run_episode(env: CombatEnv, seed: int, recorder=None) -> dict:
    env.reset(seed=seed)
    teams = (ScriptedTeam(0), ScriptedTeam(1))
    total_damage = 0.0
    total_hits = 0
    total_blocks = 0
    steps = 0
    while not env.done:
        actions = {}
        for tm in teams:
            actions.update(tm.act(env))
        _, rewards, done, info = env.step(actions)
        total_damage += float(info["damage"].sum())
        total_hits += int(info["pellet_hits"].sum())
        total_blocks += info["blocked"][0] + info["blocked"][1]
        steps += 1
        if recorder is not None:
            recorder(env)
    return {
        "sep": env.spawn_separation,
        "result": tuple(env.team_result),
        "damage": total_damage,
        "hits": total_hits,
        "blocks": total_blocks,
        "steps": steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    env = CombatEnv(seed=0)
    recorder = frames = None
    if args.video:
        import mujoco
        env.model.vis.global_.offwidth = 960
        env.model.vis.global_.offheight = 540
        renderer = mujoco.Renderer(env.model, height=540, width=960)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 0, 0.4]
        cam.distance = 8.0
        cam.azimuth = 115.0
        cam.elevation = -30.0
        frames = []

        def recorder(e):
            renderer.update_scene(e.data, camera=cam)
            frames.append(renderer.render())

    t0 = time.time()
    results = []
    for ep in range(args.episodes):
        results.append(run_episode(env, seed=1000 + ep,
                                   recorder=recorder if ep == 0 else None))
    wall = time.time() - t0

    if args.video and frames:
        import imageio
        out = Path(__file__).parent.parent / "videos" / "g2_oracle_episode.mp4"
        out.parent.mkdir(exist_ok=True)
        imageio.mimwrite(out, frames, fps=20)
        print(f"recorded {out} ({len(frames)} frames)")

    n = len(results)
    hits_pct = np.mean([r["hits"] > 0 for r in results])
    mean_damage = np.mean([r["damage"] for r in results])
    red_wins = np.mean([r["result"] == (1, -1) for r in results])
    blue_wins = np.mean([r["result"] == (-1, 1) for r in results])
    draws = 1.0 - red_wins - blue_wins
    mean_blocks = np.mean([r["blocks"] for r in results])
    seps = [r["sep"] for r in results]
    eps_s = n / wall

    print(f"\nG2 combat-signal gate — {n} oracle-vs-oracle episodes "
          f"({wall:.0f}s, {eps_s:.1f} eps/s)")
    print(f"  spawn separation      : {min(seps):.2f}–{max(seps):.2f} m")
    print(f"  episodes with hits    : {hits_pct:.0%}   (need ≥ 90%)")
    print(f"  mean damage/episode   : {mean_damage:.0f} HP (need > 30)")
    print(f"  outcomes              : red {red_wins:.0%} / blue {blue_wins:.0%} "
          f"/ draw {draws:.0%}")
    print(f"  non-draw rate         : {1-draws:.0%}  (need ≥ 60%)")
    print(f"  team win floor        : {min(red_wins, blue_wins):.0%} "
          f"(need ≥ 20%)")
    print(f"  mean shield blocks/ep : {mean_blocks:.1f}")

    checks = [
        ("hits", hits_pct >= 0.90),
        ("damage", mean_damage > 30),
        ("non-draw", (1 - draws) >= 0.60),
        ("symmetry", min(red_wins, blue_wins) >= 0.20),
    ]
    ok = all(c for _, c in checks)
    failed = [name for name, c in checks if not c]
    print(f"\nG2 gate: {'PASS' if ok else 'FAIL — ' + ', '.join(failed)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
