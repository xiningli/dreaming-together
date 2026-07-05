"""Render the trained T1 aim policy — the first learned behavior in v2.

Produces videos/t1_aim_policy.mp4: several episodes of the RL-trained
policy aiming the 2-DOF arm at randomly placed targets and firing, with
a tracer trail on the pellet (rule R7: behavior is reviewed on video,
not inferred from metrics).

Run: python tools/render_t1.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent))

import imageio
import numpy as np
import torch
import mujoco

from dreaming_together.policies.ff_policy import GaussianPolicy
from dreaming_together.training.aim_task import AimTask
from dreaming_together.envs.tank import set_arm_ctrl, muzzle_pos, muzzle_dir, PELLET_SPEED

RES_W, RES_H = 820, 480
FPS = 25
N_EPISODES = 6
POLICY_PATH = Path(__file__).parent.parent / "runs" / "t1" / "policy.pt"
OUT_PATH = Path(__file__).parent.parent / "videos" / "t1_aim_policy.mp4"


def main() -> None:
    policy = GaussianPolicy(obs_dim=3, act_dim=2)
    policy.load_state_dict(torch.load(POLICY_PATH, weights_only=True))
    policy.eval()

    task = AimTask(seed=99)
    task.model.vis.global_.offwidth = RES_W
    task.model.vis.global_.offheight = RES_H
    renderer = mujoco.Renderer(task.model, height=RES_H, width=RES_W)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 4.5
    cam.azimuth = 135.0
    cam.elevation = -22.0

    trail: list[np.ndarray] = []
    pellet_bid = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_BODY,
                                   "pellet_0")

    frames = []
    hits = 0
    for ep in range(N_EPISODES):
        obs = task.reset()
        cam.lookat[:] = (task.target + np.array([0.0, 0.0, task.target[2]])) / 2
        with torch.no_grad():
            action = torch.tanh(policy.mean_net(torch.from_numpy(obs))).numpy()
        pan, tilt = task.act_to_angles(action)
        set_arm_ctrl(task.model, task.data, "red1", pan_rad=pan, tilt_rad=tilt)
        mujoco.mj_forward(task.model, task.data)

        # brief hold so the viewer can see the aim before the shot
        for _ in range(10):
            renderer.update_scene(task.data, camera=cam)
            frames.append(renderer.render())

        origin = muzzle_pos(task.model, task.data, "red1")
        d = muzzle_dir(task.model, task.data, "red1")
        task.pm.spawn(origin, d * PELLET_SPEED, shooter="red1")

        trail.clear()
        hit = False
        step_per_frame = int(round(1.0 / FPS / 0.002))   # 20 phys steps/frame
        for _ in range(12):   # ~0.5 s of flight coverage
            for _ in range(step_per_frame):
                mujoco.mj_step(task.model, task.data)
                task.pm.step()
                if task.pm.n_active:
                    trail.append(task.data.xpos[pellet_bid].copy())
            renderer.update_scene(task.data, camera=cam)
            scene = renderer._scene
            # tracer: connect recent pellet positions
            for a, b in zip(trail[-12:-1], trail[-11:]):
                if scene.ngeom >= scene.maxgeom:
                    break
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    np.zeros(3), np.zeros(3),
                                    np.zeros(9),
                                    np.array([1.0, 0.85, 0.1, 0.9],
                                             dtype=np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                     0.008, a, b)
                scene.ngeom += 1
            frames.append(renderer.render())
            if task.pm.n_active == 0:
                break
        for h in task.pm.drain_hits():
            if h.geom_name == "paper_target_g":
                hit = True
        hits += int(hit)

    renderer.close()
    OUT_PATH.parent.mkdir(exist_ok=True)
    imageio.mimwrite(OUT_PATH, frames, fps=FPS)
    print(f"{OUT_PATH}: {len(frames)} frames, {hits}/{N_EPISODES} hits")


if __name__ == "__main__":
    main()
