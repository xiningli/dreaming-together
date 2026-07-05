"""Perception gate (G1) — rendered-POV camera tests.

Failure ledger rule R3: every camera has a rendered-POV unit test. The v0
humanoid died partly because the shield blinded its own camera and nobody
rendered what the agent saw until after training.

What these tests prove
----------------------
- Observation cameras exist with the right mounts (front on both roles,
  rear on shield only) and none is attached to the arm.
- Shotgun front cam: enemy hull visible (≥ N pixels) at spawn separation.
- Shield front cam: raised shield dominates the view (occlusion is the
  designed premise — the shield agent must rely on communication), and
  panning the arm to max opens sightlines (shield pixels drop).
- Shield rear cam: ally hull visible in column formation.
- Segmentation classes are correctly coded (red vs blue hulls distinct).

Run: python -m pytest tests/perception/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.envs.tank import set_arm_ctrl, ARM_PAN_RANGE
from dreaming_together.envs.cameras import (
    SegCamera,
    SEG_RED_HULL,
    SEG_BLUE_HULL,
    SEG_SHIELD,
)
from tests.helpers import (
    build_solo_tank,
    build_opposing_pair,
    build_column_with_enemy,
)

_SPAWN_SEPARATION = 3.0   # m — midpoint of the U(1.5, 5.0) spawn range


def _cam_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)


class TestCameraMounts:

    def test_front_cam_exists_on_both_roles(self):
        for role, prefix in [("shield", "red0"), ("shotgun", "red1")]:
            model, _ = build_solo_tank(role, prefix)
            assert _cam_id(model, f"{prefix}_front_cam") >= 0, (
                f"{role}: missing {prefix}_front_cam"
            )

    def test_rear_cam_only_on_shield(self):
        model, _ = build_solo_tank("shield", "red0")
        assert _cam_id(model, "red0_rear_cam") >= 0, "shield missing rear cam"
        model, _ = build_solo_tank("shotgun", "red1")
        assert _cam_id(model, "red1_rear_cam") < 0, (
            "shotgun must not have a rear cam (design §3.4)"
        )

    def test_obs_cameras_fixed_to_hull_not_arm(self):
        """Observation cameras must be children of the hull body (R3: no
        camera rotates with the arm — arm-mounted cameras make the policy
        chase its own arm)."""
        model, _ = build_solo_tank("shield", "red0")
        hull_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red0_hull")
        for cam in ("red0_front_cam", "red0_rear_cam"):
            cid = _cam_id(model, cam)
            assert model.cam_bodyid[cid] == hull_bid, (
                f"{cam} is mounted on body id {model.cam_bodyid[cid]}, "
                f"expected hull body {hull_bid}"
            )


class TestRenderedPOV:

    def test_shotgun_front_cam_sees_enemy_at_spawn_separation(self):
        """Enemy hull must occupy a nontrivial pixel area from the shotgun's
        front camera at spawn-range separation (kills failure F3)."""
        model, data = build_opposing_pair(
            shooter_prefix="red1", shooter_role="shotgun",
            target_prefix="blue1", target_role="shotgun",
            distance=_SPAWN_SEPARATION,
        )
        mujoco.mj_forward(model, data)
        cam = SegCamera(model)
        seg = cam.render(data, "red1_front_cam")
        cam.close()

        enemy_px = int(np.sum(seg == SEG_BLUE_HULL))
        assert enemy_px >= 4, (
            f"Enemy hull occupies only {enemy_px} px in red1_front_cam at "
            f"{_SPAWN_SEPARATION} m — agent is effectively blind to the enemy"
        )

    def test_shield_front_cam_occluded_by_raised_shield(self):
        """With the shield raised, the shield agent's front view is dominated
        by its own shield. This is the designed perceptual-occlusion premise:
        the shield agent depends on the coordinator for forward information."""
        model, data = build_opposing_pair(
            shooter_prefix="red0", shooter_role="shield",
            target_prefix="blue1", target_role="shotgun",
            distance=_SPAWN_SEPARATION,
        )
        set_arm_ctrl(model, data, "red0", pan_rad=0.0, tilt_rad=0.0)
        mujoco.mj_forward(model, data)
        cam = SegCamera(model)
        seg_raised = cam.render(data, "red0_front_cam")

        shield_frac_raised = float(np.mean(seg_raised == SEG_SHIELD))
        assert shield_frac_raised > 0.30, (
            f"Raised shield covers only {shield_frac_raised:.0%} of the shield "
            f"agent's front view — occlusion premise not satisfied"
        )

        # Panning the arm to max must open sightlines: shield coverage drops.
        set_arm_ctrl(model, data, "red0", pan_rad=ARM_PAN_RANGE[1], tilt_rad=0.0)
        mujoco.mj_forward(model, data)
        seg_open = cam.render(data, "red0_front_cam")
        cam.close()

        shield_frac_open = float(np.mean(seg_open == SEG_SHIELD))
        assert shield_frac_open < shield_frac_raised - 0.10, (
            f"Panning the shield arm to max barely changed front-view "
            f"occlusion ({shield_frac_raised:.0%} → {shield_frac_open:.0%})"
        )

    def test_shield_rear_cam_sees_ally_in_column(self):
        """In column formation the shield agent's rear camera must see the
        allied shotgun tank (design §3.4: rear cam verifies ally coverage)."""
        model, data = build_column_with_enemy(shield_distance=1.5,
                                              enemy_distance=3.0)
        mujoco.mj_forward(model, data)
        cam = SegCamera(model)
        seg = cam.render(data, "red0_rear_cam")
        cam.close()

        ally_px = int(np.sum(seg == SEG_RED_HULL))
        assert ally_px >= 4, (
            f"Ally hull occupies only {ally_px} px in red0_rear_cam in column "
            f"formation — rear camera cannot verify ally coverage"
        )

    def test_segmentation_distinguishes_teams(self):
        """Red and blue hulls must map to different class ids in the same
        frame (identity is explicit — no texture markers needed)."""
        model, data = build_column_with_enemy(shield_distance=1.5,
                                              enemy_distance=3.0)
        # Open the window so the enemy is visible past the shield.
        set_arm_ctrl(model, data, "red0", pan_rad=ARM_PAN_RANGE[1], tilt_rad=0.0)
        mujoco.mj_forward(model, data)
        cam = SegCamera(model)
        seg = cam.render(data, "red1_front_cam")
        cam.close()

        red_px  = int(np.sum(seg == SEG_RED_HULL))
        blue_px = int(np.sum(seg == SEG_BLUE_HULL))
        assert red_px > 0, "Allied red hull not visible in red1 front cam (column)"
        assert blue_px > 0, (
            "Enemy blue hull not visible in red1 front cam with window open"
        )
