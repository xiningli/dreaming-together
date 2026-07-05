"""Closed-form 2-DOF aiming expert (design §3.2).

With the elbow removed, aiming is exact and tiny: the muzzle lies on the
line through the arm mount along the arm direction, so pointing the muzzle
ray at a world target T reduces to expressing T in the hull-local frame at
the arm mount and reading off pan (azimuth) and tilt (elevation).

This is the teacher for trainability-ladder rung T0 (behavior cloning) and
the core of the scripted shotgun oracle.
"""
from __future__ import annotations

import numpy as np
import mujoco

from dreaming_together.envs.tank import (
    ARM_MOUNT_LOCAL,
    ARM_PAN_RANGE,
    ARM_TILT_RANGE,
    hull_pos,
    hull_yaw,
)


def aim_angles(model: mujoco.MjModel, data: mujoco.MjData,
               prefix: str, target_world: np.ndarray) -> tuple[float, float]:
    """Return (pan, tilt) in radians that point the muzzle ray at target_world.

    Angles are clamped to the joint ranges; targets outside the reachable
    cone get the nearest achievable angles.
    """
    hp = hull_pos(model, data, prefix)
    yaw = hull_yaw(model, data, prefix)
    c, s = np.cos(yaw), np.sin(yaw)

    mount_world = hp + np.array([c * ARM_MOUNT_LOCAL[0] - s * ARM_MOUNT_LOCAL[1],
                                 s * ARM_MOUNT_LOCAL[0] + c * ARM_MOUNT_LOCAL[1],
                                 ARM_MOUNT_LOCAL[2]])
    v = np.asarray(target_world, dtype=float) - mount_world
    # hull-local frame (yaw only; hull roll/pitch ≈ 0 on flat ground)
    vx =  c * v[0] + s * v[1]
    vy = -s * v[0] + c * v[1]
    vz =  v[2]

    pan  = float(np.arctan2(vy, vx))
    tilt = float(np.arctan2(vz, np.hypot(vx, vy)))
    pan  = float(np.clip(pan,  *ARM_PAN_RANGE))
    tilt = float(np.clip(tilt, *ARM_TILT_RANGE))
    return pan, tilt
