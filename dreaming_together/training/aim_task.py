"""Aim task — the environment for trainability-ladder rungs T0 and T1.

One shotgun tank (red1) at the origin; a 0.5 m square paper target on a
mocap body so it can be repositioned every episode without rebuilding the
model. The task: given the target position in the hull frame, set (pan,
tilt) and fire one pellet. Hit detection is the real ray-swept
ProjectileManager against real physics — this is deliberately the same hit
path the combat env will use, so a policy that trains here proves the whole
observation → action → physics → reward loop (rules R2, R4).

The observation here is the target point in the hull frame (3,), not
vision. T0/T1 validate the training loop below perception; the vision
encoder enters at Stage 0 / T2.
"""
from __future__ import annotations

import numpy as np
import mujoco

from dreaming_together.envs.tank import (
    tank_body_xml, actuator_xml, PELLET_SPEED,
    set_arm_ctrl, muzzle_pos, muzzle_dir, hull_pos, hull_yaw,
    ARM_PAN_RANGE, ARM_TILT_RANGE,
)
from dreaming_together.envs.projectiles import ProjectileManager
from tests.helpers import PHYSICS_XML_HEADER, TARGET_W, MUZZLE_Z

TARGET_HALF = TARGET_W / 2   # 0.25 m


def build_aim_range() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Shotgun tank + mocap-mounted paper target + one pooled pellet."""
    body = tank_body_xml("red1", "0.8 0.3 0.3 1", (0.0, 0.0), 0.0, "shotgun")
    acts = actuator_xml("red1")
    xml = f"""<mujoco model="aim_range">
  {PHYSICS_XML_HEADER}
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="10 10 0.1" friction="0.5 0.01 0.001" rgba="0.6 0.6 0.6 1"/>
    {body}
    <body name="paper_target" mocap="true" pos="2.5 0 {MUZZLE_Z}">
      <geom name="paper_target_g" type="box"
            size="0.01 {TARGET_HALF} {TARGET_HALF}"
            rgba="1 0.9 0.7 1" contype="1" conaffinity="7"/>
    </body>
    <body name="pellet_0" pos="0 0 100">
      <freejoint name="pellet_0_jnt"/>
      <geom name="pellet_0_g" type="sphere" size="0.01" mass="0.005"
            contype="2" conaffinity="7" rgba="1 0.8 0 1"/>
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class AimTask:
    """Single-shot aiming episodes with randomized target placement."""

    # target sampling ranges (reachable cone, design §3.2 grid)
    DIST_RANGE    = (1.8, 3.0)
    BEARING_RANGE = (-np.radians(40), np.radians(40))
    DZ_RANGE      = (-0.15, 0.30)

    def __init__(self, seed: int = 0):
        self.model, self.data = build_aim_range()
        self.pm = ProjectileManager(self.model, self.data)
        self.rng = np.random.default_rng(seed)
        self.target = np.zeros(3)

    def reset(self) -> np.ndarray:
        """Place the target at a random reachable pose; return the obs:
        the target point in the hull frame, scaled by 1/3."""
        mujoco.mj_resetData(self.model, self.data)
        self.pm = ProjectileManager(self.model, self.data)
        dist    = self.rng.uniform(*self.DIST_RANGE)
        bearing = self.rng.uniform(*self.BEARING_RANGE)
        dz      = self.rng.uniform(*self.DZ_RANGE)
        self.target = np.array([dist * np.cos(bearing),
                                dist * np.sin(bearing),
                                MUZZLE_Z + dz])
        mid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                "paper_target")
        self.data.mocap_pos[self.model.body_mocapid[mid]] = self.target
        # rotate the board so its face stays roughly head-on to the tank
        self.data.mocap_quat[self.model.body_mocapid[mid]] = [
            np.cos(bearing / 2), 0, 0, np.sin(bearing / 2)]
        mujoco.mj_forward(self.model, self.data)
        return self.obs()

    def obs(self) -> np.ndarray:
        hp = hull_pos(self.model, self.data, "red1")
        yaw = hull_yaw(self.model, self.data, "red1")
        c, s = np.cos(yaw), np.sin(yaw)
        v = self.target - hp
        local = np.array([c * v[0] + s * v[1],
                          -s * v[0] + c * v[1],
                          v[2]])
        return (local / 3.0).astype(np.float32)

    def act_to_angles(self, action: np.ndarray) -> tuple[float, float]:
        """Map normalized action in [-1, 1]^2 to (pan, tilt) radians."""
        a = np.clip(action, -1.0, 1.0)
        pan  = ARM_PAN_RANGE[0]  + (a[0] + 1) / 2 * (ARM_PAN_RANGE[1]  - ARM_PAN_RANGE[0])
        tilt = ARM_TILT_RANGE[0] + (a[1] + 1) / 2 * (ARM_TILT_RANGE[1] - ARM_TILT_RANGE[0])
        return float(pan), float(tilt)

    def angles_to_act(self, pan: float, tilt: float) -> np.ndarray:
        a0 = 2 * (pan  - ARM_PAN_RANGE[0])  / (ARM_PAN_RANGE[1]  - ARM_PAN_RANGE[0])  - 1
        a1 = 2 * (tilt - ARM_TILT_RANGE[0]) / (ARM_TILT_RANGE[1] - ARM_TILT_RANGE[0]) - 1
        return np.array([a0, a1], dtype=np.float32)

    def fire(self, action: np.ndarray) -> tuple[float, bool]:
        """Set the arm, fire one pellet, run physics until resolution.

        Returns (shaped_reward, hit):
          hit            — ray-swept pellet crossed paper_target_g
          shaped_reward  — hit bonus + dense aim shaping (−miss distance).
                           The shaping is linear and unsaturated so the
                           gradient is informative everywhere, including at
                           random init (a max(0, 1−miss/0.5) variant stalled
                           T1 at 14% because misses > 0.5 m gave zero signal).
        """
        pan, tilt = self.act_to_angles(action)
        set_arm_ctrl(self.model, self.data, "red1", pan_rad=pan, tilt_rad=tilt)
        mujoco.mj_forward(self.model, self.data)

        origin = muzzle_pos(self.model, self.data, "red1")
        d = muzzle_dir(self.model, self.data, "red1")
        # dense shaping from the muzzle ray's closest approach to the target
        v = self.target - origin
        along = max(float(np.dot(v, d)), 1e-6)
        miss = float(np.linalg.norm(v - along * d))
        shaped = -min(miss, 3.0)

        self.pm.spawn(origin, d * PELLET_SPEED, shooter="red1")
        hit = False
        for _ in range(120):   # 240 ms — enough for 3 m of flight + margin
            mujoco.mj_step(self.model, self.data)
            self.pm.step()
            if self.pm.n_active == 0:
                break
        for h in self.pm.drain_hits():
            if h.geom_name == "paper_target_g":
                hit = True
        return shaped + (2.0 if hit else 0.0), hit
