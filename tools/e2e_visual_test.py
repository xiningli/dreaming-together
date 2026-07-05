"""End-to-end visual engine test — scripted 2v2 combat episode on video.

Everything the engine has, exercised together in one choreographed episode
(precursor of the G2 combat-signal gate):

  Phase 1 (advance)   Both teams drive toward each other, shields raised.
  Phase 2 (siege)     Teams halt in range. Blue shotgun volleys; red's
                      shield blocks the corridor (pellets visibly stop).
  Phase 3 (window)    Red coordinator cue: red shield opens the window
                      (max pan), red shotgun fires 8-pellet cones through
                      it at blue1 via the IK expert.
  Phase 4 (result)    HP is deducted per ray-swept hull hit (6 HP/pellet);
                      one agent at 0 HP = team loss = episode over.

Output: videos/e2e_scripted_combat.mp4 (tracer rounds, HP bars above
hulls) plus an event log on stdout. Review the video (rule R7).

Run: python tools/e2e_visual_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).parent.parent))

import imageio
import numpy as np
import mujoco

from dreaming_together.envs.arena import build_combat_arena
from dreaming_together.envs.projectiles import ProjectileManager, sample_pellet_directions
from dreaming_together.envs.tank import (
    N_PELLETS, PELLET_SPEED, ARM_PAN_RANGE,
    set_track_ctrl, set_arm_target, muzzle_pos, muzzle_dir,
    hull_pos, hull_yaw, window_quality,
)
from tools.ik_expert import aim_angles

RES_W, RES_H = 960, 540
DT_POLICY = 0.05                 # 50 ms control period
SUBSTEPS = 25                    # 25 × 2 ms
FPS = 20                         # one frame per control step
EPISODE_CAP_S = 20.0
FIRE_PERIOD_S = 1.2
HP_MAX = 100
HP_PER_PELLET = 6
ENGAGE_DIST = 2.4                # halt distance
WINDOW_CUE_S = 7.0               # red coordinator opens the window here

TEAMS = {"red": ("red0", "red1"), "blue": ("blue0", "blue1")}
OUT_PATH = Path(__file__).parent.parent / "videos" / "e2e_scripted_combat.mp4"

# Column formation for red (shield screens the shotgun); blue side-by-side.
SPAWNS = {
    "red0":  ((-2.0,  0.0), 0.0),      # shield in front
    "red1":  ((-3.1,  0.0), 0.0),      # shotgun behind it
    "blue0": (( 3.0, -0.5), np.pi),
    "blue1": (( 3.0,  0.5), np.pi),
}

# Scripted advance waypoints keep the red column intact (shield screening
# the shotgun) and the hull yaws stable; free mutual pursuit deformed the
# formation and shots clipped the own-shield corner mid-drive.
# red0 sits laterally offset from red1's firing lane: the shield plate
# (1.2 m wide) still covers the corridor when raised, but red1's open-window
# shots clear red0's own hull (W_q counts only the shield geom as a blocker,
# so an on-axis column lets red1 shoot its own shield agent in the back —
# observed in this test's first passing run).
WAYPOINTS = {
    "red0":  np.array([-0.3, 0.55]),
    "red1":  np.array([-1.4, 0.0]),
    "blue0": np.array([ 1.2, -0.5]),
    "blue1": np.array([ 1.2,  0.35]),
}


class Game:
    """Minimal scripted game layer over the physics engine."""

    def __init__(self):
        self.model, self.data = build_combat_arena(SPAWNS)
        self.pm = ProjectileManager(self.model, self.data)
        self.hp = {p: HP_MAX for p in SPAWNS}
        self.cooldown = {"red1": 0.6, "blue1": 0.2}   # staggered first shots
        self.t = 0.0
        self.log: list[str] = []
        self.blocked = 0
        self.hull_hits = 0

    # -- scripted behaviors ------------------------------------------------
    def _drive_toward(self, prefix: str, target_xy: np.ndarray,
                      halt_dist: float) -> None:
        pos = hull_pos(self.model, self.data, prefix)[:2]
        yaw = hull_yaw(self.model, self.data, prefix)
        v = target_xy - pos
        dist = float(np.linalg.norm(v))
        if dist < halt_dist:
            set_track_ctrl(self.model, self.data, prefix, 0.0, 0.0)
            return
        bearing = np.arctan2(v[1], v[0]) - yaw
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        fwd = 0.6 * np.cos(bearing)
        turn = np.clip(1.5 * bearing, -0.5, 0.5)
        set_track_ctrl(self.model, self.data, prefix,
                       fwd - turn, fwd + turn)

    def _nearest_opponent(self, prefix: str) -> str:
        team = "red" if prefix.startswith("red") else "blue"
        opps = TEAMS["blue" if team == "red" else "red"]
        pos = hull_pos(self.model, self.data, prefix)[:2]
        alive = [o for o in opps if self.hp[o] > 0] or list(opps)
        return min(alive, key=lambda o: np.linalg.norm(
            hull_pos(self.model, self.data, o)[:2] - pos))

    def _fire(self, prefix: str) -> None:
        origin = muzzle_pos(self.model, self.data, prefix)
        base = muzzle_dir(self.model, self.data, prefix)
        dirs = sample_pellet_directions(base, N_PELLETS,
                                        seed=int(self.t * 1000) % 2 ** 31)
        for d in dirs:
            self.pm.spawn(origin, d * PELLET_SPEED, shooter=prefix)
        self.log.append(f"[{self.t:5.2f}s] {prefix} FIRES "
                        f"({N_PELLETS} pellets)")

    def control(self) -> None:
        """One 50 ms scripted decision for every agent."""
        # Window timing, the coordination pattern the coordinator must
        # eventually learn: open just before the shotgun is ready, close
        # right after the volley. A window left standing open exposes the
        # shield agent (blue shredded red0 in an earlier run of this test).
        window_open = (self.t >= WINDOW_CUE_S
                       and self.cooldown["red1"] <= 0.35)

        # shields: red0 screens the corridor; opens on the coordinator cue
        set_arm_target(self.model, self.data, "red0",
                       pan_rad=ARM_PAN_RANGE[1] if window_open else 0.0,
                       tilt_rad=0.0)
        set_arm_target(self.model, self.data, "blue0", pan_rad=0.0,
                       tilt_rad=0.0)

        # drive: advance to scripted waypoints (keeps the column intact)
        for prefix in SPAWNS:
            if self.hp[prefix] <= 0:
                set_track_ctrl(self.model, self.data, prefix, 0.0, 0.0)
                continue
            self._drive_toward(prefix, WAYPOINTS[prefix], 0.15)

        # shotguns: track the target with the IK expert, fire on cooldown
        for shooter in ("red1", "blue1"):
            if self.hp[shooter] <= 0:
                continue
            opp = self._nearest_opponent(shooter)
            aim_at = hull_pos(self.model, self.data, opp) + [0, 0, 0.15]
            pan, tilt = aim_angles(self.model, self.data, shooter, aim_at)
            set_arm_target(self.model, self.data, shooter, pan, tilt)

            self.cooldown[shooter] -= DT_POLICY
            in_range = np.linalg.norm(
                hull_pos(self.model, self.data, opp)[:2]
                - hull_pos(self.model, self.data, shooter)[:2]) < 3.0
            if self.cooldown[shooter] <= 0 and in_range:
                # red1 holds fire until the coordinator cue AND the window
                # is genuinely open (W_q from its own muzzle cone)
                if shooter == "red1":
                    if not window_open:
                        continue
                    wq = window_quality(self.model, self.data,
                                        "red1", "red0")
                    if wq < 0.5:
                        continue
                self._fire(shooter)
                self.cooldown[shooter] = FIRE_PERIOD_S

    def resolve_hits(self) -> None:
        for h in self.pm.drain_hits():
            if h.geom_name.endswith("_hull_g"):
                victim = h.geom_name[:-len("_hull_g")]
                self.hp[victim] = max(0, self.hp[victim] - HP_PER_PELLET)
                self.hull_hits += 1
                self.log.append(
                    f"[{self.t:5.2f}s] {h.shooter} pellet HIT {victim} "
                    f"hull → HP {self.hp[victim]}")
                if self.hp[victim] == 0:
                    self.log.append(f"[{self.t:5.2f}s] {victim} ELIMINATED "
                                    f"— team {'red' if victim.startswith('red') else 'blue'} loses")
            elif h.geom_name.endswith("_shield_g"):
                self.blocked += 1
                self.log.append(
                    f"[{self.t:5.2f}s] {h.shooter} pellet BLOCKED by "
                    f"{h.geom_name}")

    def done(self) -> bool:
        dead_team = any(self.hp[p] <= 0 for p in SPAWNS)
        return dead_team or self.t >= EPISODE_CAP_S


def hp_bar_overlay(scene: mujoco.MjvScene, game: Game) -> None:
    """Draw a floating HP bar above each living hull."""
    for prefix in SPAWNS:
        frac = game.hp[prefix] / HP_MAX
        pos = hull_pos(game.model, game.data, prefix) + [0, 0, 1.35]
        if scene.ngeom >= scene.maxgeom - 1:
            return
        g = scene.geoms[scene.ngeom]
        rgba = np.array([1 - frac, frac, 0.1, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                            np.array([0.02 + 0.28 * frac, 0.05, 0.02]),
                            pos, np.eye(3).flatten(), rgba)
        scene.ngeom += 1


def tracer_overlay(scene: mujoco.MjvScene, trails: dict) -> None:
    for pts in trails.values():
        for a, b in zip(pts[:-1], pts[1:]):
            if scene.ngeom >= scene.maxgeom:
                return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.zeros(9),
                                np.array([1.0, 0.85, 0.1, 0.9],
                                         dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                 0.008, a, b)
            scene.ngeom += 1


def main() -> None:
    game = Game()
    game.model.vis.global_.offwidth = RES_W
    game.model.vis.global_.offheight = RES_H
    renderer = mujoco.Renderer(game.model, height=RES_H, width=RES_W)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.4]
    cam.distance = 7.5
    cam.azimuth = 115.0
    cam.elevation = -30.0

    pellet_bids = {n: mujoco.mj_name2id(game.model, mujoco.mjtObj.mjOBJ_BODY, n)
                   for n in game.pm._names}
    trails: dict[str, list] = {}

    frames = []
    while not game.done():
        game.control()
        for _ in range(SUBSTEPS):
            mujoco.mj_step(game.model, game.data)
            game.pm.step()
            for name, bid in pellet_bids.items():
                p = game.data.xpos[bid].copy()
                if p[2] < 50:   # active (not parked)
                    trails.setdefault(name, []).append(p)
                    trails[name] = trails[name][-10:]
                elif name in trails:
                    trails[name] = trails[name][1:] or None
                    if not trails[name]:
                        del trails[name]
        game.resolve_hits()
        game.t += DT_POLICY

        renderer.update_scene(game.data, camera=cam)
        tracer_overlay(renderer._scene, trails)
        hp_bar_overlay(renderer._scene, game)
        frames.append(renderer.render())

    renderer.close()
    OUT_PATH.parent.mkdir(exist_ok=True)
    imageio.mimwrite(OUT_PATH, frames, fps=FPS)

    print("\n".join(game.log))
    print(f"\n{OUT_PATH}")
    print(f"episode length: {game.t:.1f} s, {len(frames)} frames")
    print(f"shield blocks: {game.blocked}, hull hits: {game.hull_hits}")
    print(f"final HP: {game.hp}")
    ok = game.blocked > 0 and game.hull_hits > 0 and any(
        v == 0 for v in game.hp.values())
    print(f"E2E visual test {'PASS' if ok else 'INCOMPLETE'}: "
          f"blocks seen, hull damage dealt, elimination reached"
          if ok else
          "E2E visual test INCOMPLETE — expected blocks + hits + elimination")


if __name__ == "__main__":
    main()
