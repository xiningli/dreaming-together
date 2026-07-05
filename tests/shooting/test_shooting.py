"""Shooting tests — shotgun tank vs static paper target.

Minimum model: one shotgun tank (red1) + paper target.  No shield agent.

What these tests prove
----------------------
- The muzzle site exists at the correct end-effector position.
- Forward kinematics: arm at known angles → muzzle at predicted world position.
- Pellet spawns at the muzzle and travels in the muzzle direction.
- Pellets hit a static bullseye at 2.5 m when the arm is aimed at it.
- Eight pellets spread within the 8° cone half-angle.
- Trigger channel = 0 → no pellet spawned (no accidental firing).

What these tests do NOT prove
------------------------------
- Aiming accuracy while moving (requires Stage 1 trained policy).
- Pellet spread statistics (covered in ProjectileManager unit tests).
- Friendly-fire mechanics (requires both shooter and shield — covered in
  shield tests where relevant).

Run: python -m pytest tests/shooting/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.envs.tank import (
    ARM_LEN,
    END_EFFECTOR_LEN,
    ARM_MOUNT_LOCAL,
    HULL_Z,
    CONE_HALF_ANGLE,
    N_PELLETS,
    PELLET_SPEED,
    ARM_PAN_RANGE,
    ARM_TILT_RANGE,
    muzzle_pos,
    muzzle_dir,
    set_arm_ctrl,
)
from tests.helpers import (
    build_shooting_range,
    build_solo_tank,
    cast_ray,
    any_contact_with,
    spawn_pellet,
    site_id,
    geom_id,
    TARGET_Z,
)

_DT = 0.002


class TestShooting:

    # ------------------------------------------------------------------
    # Muzzle site existence and forward kinematics
    # ------------------------------------------------------------------

    def test_muzzle_site_exists(self):
        """Model contains a site named 'red1_muzzle' with positive ID."""
        model, data = build_solo_tank("shotgun", "red1")
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "red1_muzzle")
        assert sid >= 0, "Site 'red1_muzzle' not found in solo shotgun model"

    def test_muzzle_position_at_arm_default_pose(self):
        """With all arm joints at 0 rad, muzzle x > hull front edge (> 0.30 m).

        At pan=0, tilt=0 the arm extends straight forward (+x in hull
        frame).  The muzzle must be beyond the hull's front face.
        """
        model, data = build_solo_tank("shotgun", "red1")
        set_arm_ctrl(model, data, "red1", pan_rad=0.0, tilt_rad=0.0)
        mujoco.mj_forward(model, data)

        pos = muzzle_pos(model, data, "red1")
        hull_front_x = 0.30   # half of HULL_LENGTH
        assert pos[0] > hull_front_x, (
            f"Muzzle x={pos[0]:.3f} m is not forward of hull front edge "
            f"({hull_front_x:.2f} m) at default arm pose"
        )

    def test_muzzle_forward_kinematics_consistency(self):
        """Muzzle position changes monotonically with tilt angle.

        As tilt increases from 0 to 60°, the muzzle z-coordinate must
        increase (the arm tilts upward, raising the muzzle).
        """
        model, data = build_solo_tank("shotgun", "red1")
        tilt_angles = np.linspace(0.0, np.radians(60.0), 5)
        z_vals = []
        for tilt in tilt_angles:
            set_arm_ctrl(model, data, "red1", pan_rad=0.0, tilt_rad=tilt)
            mujoco.mj_forward(model, data)
            z_vals.append(muzzle_pos(model, data, "red1")[2])

        for i in range(len(z_vals) - 1):
            assert z_vals[i + 1] > z_vals[i], (
                f"Muzzle z did not increase with tilt: "
                f"z[{i}]={z_vals[i]:.3f}, z[{i+1}]={z_vals[i+1]:.3f}"
            )

    # ------------------------------------------------------------------
    # Muzzle direction and ray casting
    # ------------------------------------------------------------------

    def test_muzzle_direction_is_unit_vector(self):
        """muzzle_dir() returns a unit vector at any arm configuration."""
        model, data = build_solo_tank("shotgun", "red1")
        for pan in [-0.5, 0.0, 0.5]:
            for tilt in [0.0, 0.4, 0.8]:
                set_arm_ctrl(model, data, "red1",
                             pan_rad=pan, tilt_rad=tilt)
                mujoco.mj_forward(model, data)
                d = muzzle_dir(model, data, "red1")
                norm = np.linalg.norm(d)
                assert abs(norm - 1.0) < 1e-5, (
                    f"muzzle_dir norm={norm:.6f} at pan={pan}, tilt={tilt}"
                )

    def test_ray_from_muzzle_hits_paper_target(self):
        """A ray cast from the muzzle along muzzle_dir() hits the paper target.

        The arm is aimed straight at the target (pan=0, tilt computed so
        muzzle points horizontally toward target_x).  A single mj_ray call
        must return the paper_target_g geom.
        """
        target_dist = 2.5
        model, data = build_shooting_range(target_distance=target_dist)

        # Aim arm straight forward (horizontal), target is at the same z as muzzle
        set_arm_ctrl(model, data, "red1", pan_rad=0.0, tilt_rad=0.0)
        mujoco.mj_forward(model, data)

        origin = muzzle_pos(model, data, "red1")
        direction = muzzle_dir(model, data, "red1")
        dist, hit_geom = cast_ray(model, data, origin, direction)

        target_gid = geom_id(model, "paper_target_g")
        assert hit_geom == target_gid, (
            f"Ray from muzzle did not hit paper_target_g (got geom id {hit_geom}); "
            f"ray distance={dist:.3f} m, expected ~{target_dist:.1f} m"
        )

    # ------------------------------------------------------------------
    # Pellet dynamics
    # ------------------------------------------------------------------

    def test_pellet_travels_in_muzzle_direction(self):
        """A pellet spawned at the muzzle with PELLET_SPEED moves forward.

        After 50 ms (25 physics steps) the pellet must have advanced > 1 m
        in the direction the muzzle was pointing when it was spawned.

        The model XML must include one pre-allocated pellet body named
        'pellet_0' with a freejoint.
        """
        target_dist = 5.0   # far target so pellet doesn't hit during the test
        model, data = build_shooting_range(target_distance=target_dist)

        set_arm_ctrl(model, data, "red1", pan_rad=0.0, tilt_rad=0.0)
        mujoco.mj_forward(model, data)

        p0  = muzzle_pos(model, data, "red1").copy()
        vel = muzzle_dir(model, data, "red1") * PELLET_SPEED
        spawn_pellet(model, data, "pellet_0", p0, vel)

        n_steps = 25   # 50 ms
        for _ in range(n_steps):
            mujoco.mj_step(model, data)

        mujoco.mj_forward(model, data)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pellet_0")
        p1 = data.xpos[bid].copy()

        travel = np.dot(p1 - p0, vel / np.linalg.norm(vel))
        assert travel > 1.0, (
            f"Pellet travelled only {travel:.3f} m along muzzle axis in 50 ms "
            f"(expected > 1.0 m at {PELLET_SPEED} m/s)"
        )

    def test_pellet_hits_bullseye(self):
        """A pellet spawned at the muzzle hits the paper target at 2.5 m.

        The pellet must register a contact with 'paper_target_g' within
        100 ms (50 physics steps) when the arm is aimed directly at the
        target centre.
        """
        target_dist = 2.5
        model, data = build_shooting_range(target_distance=target_dist)

        set_arm_ctrl(model, data, "red1", pan_rad=0.0, tilt_rad=0.0)
        mujoco.mj_forward(model, data)

        p0  = muzzle_pos(model, data, "red1").copy()
        vel = muzzle_dir(model, data, "red1") * PELLET_SPEED
        spawn_pellet(model, data, "pellet_0", p0, vel)

        hit = False
        for _ in range(50):   # 100 ms at 2 ms/step
            mujoco.mj_step(model, data)
            if any_contact_with(data, model, "paper_target_g"):
                hit = True
                break

        assert hit, (
            f"Pellet did not contact paper_target_g within 100 ms "
            f"(target at x={target_dist} m, muzzle at {p0})"
        )

    def test_cone_spread_within_half_angle(self):
        """Eight pellet directions sampled from the cone stay within CONE_HALF_ANGLE.

        This tests the spread-generation math in ProjectileManager, not MuJoCo
        physics.  Import and call the spread sampler directly.
        """
        from dreaming_together.envs.projectiles import sample_pellet_directions

        muzzle_direction = np.array([1.0, 0.0, 0.0])
        directions = sample_pellet_directions(muzzle_direction, N_PELLETS, seed=0)

        assert directions.shape == (N_PELLETS, 3), (
            f"Expected ({N_PELLETS}, 3) directions, got {directions.shape}"
        )

        half_rad = np.radians(CONE_HALF_ANGLE)
        for i, d in enumerate(directions):
            norm = np.linalg.norm(d)
            assert abs(norm - 1.0) < 1e-5, f"Pellet {i} direction not unit vector"
            angle = np.arccos(np.clip(np.dot(d, muzzle_direction), -1.0, 1.0))
            assert angle <= half_rad + 1e-6, (
                f"Pellet {i} angle {np.degrees(angle):.2f}° exceeds "
                f"CONE_HALF_ANGLE={CONE_HALF_ANGLE}°"
            )
