"""Render the 'communication matters' demo pair.

Two episodes of the certified condition-C stack (diffusion + language,
sample-and-select) against the frozen elite opponent, SAME spawn seed:
one with the coordinator speaking (demo/comm_on.mp4), one with z_g zeroed
(demo/comm_off.mp4). Seeds are searched until the pair tells the true
statistical story (coordinated team wins, silenced team loses) — which is
the 0.92-vs-0.45 G6 result made visible.

Run: python tools/render_comm_demo.py
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

from PIL import Image, ImageDraw, ImageFont

from dreaming_together.coordinators.vocab import TOKENS
from dreaming_together.training.stage2_diffusion import (
    make_listener, Scorer, sas_act)
from dreaming_together.training.stage2_coordination import (
    make_coordinator, coord_state, DT_COORD_STEPS, Z_DIM)

FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 22)
FONT_SM = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 17)


def annotate(frame, banner, banner_color, message):
    """Top banner (who this is + win rate) and bottom bar (the live
    coordinator message, or the silenced notice)."""
    img = Image.fromarray(frame)
    out = Image.new("RGB", (img.width, img.height + 76), (10, 10, 14))
    out.paste(img, (0, 38))
    d = ImageDraw.Draw(out)
    d.text((12, 7), banner, fill=banner_color, font=FONT)
    d.text((12, img.height + 44), message, fill=(255, 225, 120),
           font=FONT_SM)
    return np.asarray(out)

ROOT = Path(__file__).parent.parent
RUN = ROOT / "runs" / "stage2_C_diff_s1"
OUT = ROOT / "demo" / "03_communication_matters"


def play(r0, q0, r1, q1, coord, seed, mode, record=False):
    import mujoco
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import EliteScriptedTeam
    from dreaming_together.envs.tank import hull_pos
    env = CombatEnv(seed=0, privileged_obs=True)
    frames = []
    renderer = cam = None
    if record:
        env.model.vis.global_.offwidth = 960
        env.model.vis.global_.offheight = 544
        renderer = mujoco.Renderer(env.model, height=544, width=960)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 0, 0.4]; cam.distance = 8.0
        cam.azimuth = 115.0; cam.elevation = -30.0
    blue = EliteScriptedTeam(1)
    env.reset(seed=seed)
    z = np.zeros(Z_DIM, dtype=np.float32)
    step = 0
    msg = ""
    with torch.no_grad():
        while not env.done:
            if step % DT_COORD_STEPS == 0 and mode == "on":
                zt, toks, *_ = coord(torch.from_numpy(
                    coord_state(env)).unsqueeze(0), sample=False)
                z = zt[0].numpy()
                words = [TOKENS[int(i)] for i in toks[0] if TOKENS[int(i)] != "PAD"]
                msg = "coordinator \u2192 " + " ".join(words)
            actions = blue.act(env)
            for p, d, q in (("red0", r0, q0), ("red1", r1, q1)):
                o = np.concatenate([env.obs(p), z]).astype(np.float32)
                a, _ = sas_act(d, q, o, det=True)
                actions[p] = a
            env.step(actions)
            step += 1
            if record:
                renderer.update_scene(env.data, camera=cam)
                scene = renderer._scene
                for p in ("red0", "red1", "blue0", "blue1"):
                    frac = env.hp[p] / 100.0
                    if scene.ngeom >= scene.maxgeom:
                        break
                    g = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(
                        g, mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([0.02 + 0.28 * frac, 0.05, 0.02]),
                        hull_pos(env.model, env.data, p) + [0, 0, 1.35],
                        np.eye(3).flatten(),
                        np.array([1 - frac, frac, 0.1, 0.9], np.float32))
                    scene.ngeom += 1
                if mode == "on":
                    banner = ("LANGUAGE CHANNEL LIVE \u2014 "
                              "this team wins 92% of episodes")
                    bcol = (120, 255, 140)
                    bottom = msg
                else:
                    banner = ("CHANNEL SILENCED (z_g = 0) \u2014 "
                              "same team wins only 45%")
                    bcol = (255, 120, 110)
                    bottom = "no messages \u2014 shield cannot know when its shotgun is ready"
                frames.append(annotate(renderer.render(), banner, bcol,
                                       bottom))
    if record:
        renderer.close()
    win = tuple(env.team_result) == (1, -1)
    return win, frames, env.t


def main() -> None:
    torch.manual_seed(0)
    r0, r1 = make_listener(), make_listener()
    q0, q1 = Scorer(), Scorer()
    coord = make_coordinator("C")
    r0.load_state_dict(torch.load(RUN / "r0_final.pt", weights_only=True))
    r1.load_state_dict(torch.load(RUN / "r1_final.pt", weights_only=True))
    q0.load_state_dict(torch.load(RUN / "q0_final.pt", weights_only=True))
    q1.load_state_dict(torch.load(RUN / "q1_final.pt", weights_only=True))
    coord.load_state_dict(torch.load(RUN / "coord_final.pt",
                                     weights_only=True))
    for m in (r0, r1, q0, q1, coord):
        m.eval()

    # find a seed where the pair tells the statistical truth
    chosen = None
    for seed in range(970_000, 970_030):
        w_on, _, _ = play(r0, q0, r1, q1, coord, seed, "on")
        w_off, _, _ = play(r0, q0, r1, q1, coord, seed, "off")
        print(f"seed {seed}: on={'WIN' if w_on else 'loss'} "
              f"off={'win' if w_off else 'LOSS'}")
        if w_on and not w_off:
            chosen = seed
            break
    assert chosen is not None, "no representative seed in 30 tries"

    OUT.mkdir(parents=True, exist_ok=True)
    for mode, name in (("on", "comm_on.mp4"), ("off", "comm_off.mp4")):
        win, frames, t = play(r0, q0, r1, q1, coord, chosen, mode,
                              record=True)
        imageio.mimwrite(OUT / name, frames, fps=20)
        print(f"{name}: seed {chosen}, win={win}, {t:.1f}s, "
              f"{len(frames)} frames")


if __name__ == "__main__":
    main()
