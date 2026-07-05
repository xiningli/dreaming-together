"""IK expert accuracy (G0, design §3.2): muzzle ray within 2 cm of target.

Sweeps a grid of reachable targets (range 1.5–3 m, bearing ±45°, elevation
around muzzle height), computes closed-form (pan, tilt), applies them, and
measures the perpendicular miss distance from the target point to the
muzzle ray. Also checks accuracy from a rotated hull, since the expert
must work in arbitrary world poses.

Run: python -m pytest tests/shooting/test_ik_expert.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.envs.tank import (
    set_arm_ctrl, muzzle_pos, muzzle_dir,
)
from tools.ik_expert import aim_angles
from tests.helpers import build_solo_tank, MUZZLE_Z

_MISS_TOL = 0.02   # m — acceptance from the design doc


def _miss_distance(model, data, prefix, target):
    origin = muzzle_pos(model, data, prefix)
    d = muzzle_dir(model, data, prefix)
    v = target - origin
    along = float(np.dot(v, d))
    if along <= 0:
        return float(np.linalg.norm(v))   # target behind muzzle — full miss
    return float(np.linalg.norm(v - along * d))


class TestIKExpert:

    def test_grid_of_reachable_targets_within_2cm(self):
        model, data = build_solo_tank("shotgun", "red1")
        worst = 0.0
        for dist in (1.5, 2.0, 2.5, 3.0):
            for bearing in np.radians([-45, -20, 0, 20, 45]):
                for dz in (-0.15, 0.0, 0.3):
                    target = np.array([dist * np.cos(bearing),
                                       dist * np.sin(bearing),
                                       MUZZLE_Z + dz])
                    pan, tilt = aim_angles(model, data, "red1", target)
                    set_arm_ctrl(model, data, "red1", pan_rad=pan, tilt_rad=tilt)
                    mujoco.mj_forward(model, data)
                    miss = _miss_distance(model, data, "red1", target)
                    worst = max(worst, miss)
                    assert miss < _MISS_TOL, (
                        f"IK miss {miss*100:.1f} cm at dist={dist}, "
                        f"bearing={np.degrees(bearing):.0f}°, dz={dz}"
                    )

    def test_accuracy_from_rotated_hull(self):
        """The expert must aim correctly regardless of hull yaw."""
        model, data = build_solo_tank("shotgun", "red1")
        # rotate the hull in place to yaw = 140°
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red1_hull")
        jid = model.body_jntadr[bid]
        qadr = model.jnt_qposadr[jid]
        yaw = np.radians(140.0)
        data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(model, data)

        # target 2.2 m away, 25° left of the new heading
        b = yaw + np.radians(25.0)
        target = np.array([2.2 * np.cos(b), 2.2 * np.sin(b), MUZZLE_Z + 0.1])
        pan, tilt = aim_angles(model, data, "red1", target)
        set_arm_ctrl(model, data, "red1", pan_rad=pan, tilt_rad=tilt)
        mujoco.mj_forward(model, data)
        miss = _miss_distance(model, data, "red1", target)
        assert miss < _MISS_TOL, f"IK miss {miss*100:.1f} cm from rotated hull"
