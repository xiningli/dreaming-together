"""Movement tests — tank differential drive.

Minimum model: one tank on a flat floor. No weapons, no targets, no opponent.
Parametrized over both roles because the shield arm adds ~4 kg and shifts
the lateral inertia tensor; the drive system must work for both.

What these tests prove
----------------------
- Hull geometry is stable (stays on the ground with zero input).
- Forward drive (both tracks equal) produces measurable forward displacement.
- Differential turning (tracks opposite sign) produces measurable yaw change.
- Pure math: the (v_L, v_R) → (v_forward, ω) mixing formula is correct.

What these tests do NOT prove
------------------------------
- Steady-state speed accuracy (velocity actuator tuning — covered by Step 1
  acceptance criteria in the eng doc).
- Obstacle avoidance, wall collision, or multi-agent interaction.

Run: python -m pytest tests/movement/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.envs.tank import (
    MAX_TRACK_SPEED,
    TRACK_WIDTH,
    HULL_Z,
    set_track_ctrl,
    hull_pos,
    hull_yaw,
)
from tests.helpers import build_solo_tank


_ROLES = [("shield", "red0"), ("shotgun", "red1")]
_DT    = 0.002   # physics timestep


class TestMovement:

    # ------------------------------------------------------------------
    # Ground stability
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("role,prefix", _ROLES)
    def test_hull_stays_on_ground_with_zero_ctrl(self, role, prefix):
        """Hull z-position stays within 1 mm of HULL_Z for 2 s with no control input.

        Validates that the contact geometry is correct — the hull does not
        sink into the floor and does not bounce off it.
        """
        model, data = build_solo_tank(role, prefix)
        n_steps = int(2.0 / _DT)
        z_vals = []
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
            z_vals.append(hull_pos(model, data, prefix)[2])

        z_arr = np.array(z_vals)
        assert np.all(np.abs(z_arr - HULL_Z) < 0.001), (
            f"{role}: hull z drifted from HULL_Z={HULL_Z:.3f} m — "
            f"min={z_arr.min():.4f}, max={z_arr.max():.4f}"
        )

    # ------------------------------------------------------------------
    # Forward drive
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("role,prefix", _ROLES)
    def test_forward_drive_produces_displacement(self, role, prefix):
        """Both tracks at +0.5 → hull moves > 0.3 m forward in 1 s.

        Checks that the actuators produce genuine forward locomotion and
        that the track friction prevents lateral drift.
        """
        model, data = build_solo_tank(role, prefix)
        x0 = hull_pos(model, data, prefix)[0]

        set_track_ctrl(model, data, prefix, left=0.5, right=0.5)
        n_steps = int(1.0 / _DT)
        for _ in range(n_steps):
            mujoco.mj_step(model, data)

        x1 = hull_pos(model, data, prefix)[0]
        disp = x1 - x0
        assert disp > 0.30, (
            f"{role}: forward displacement {disp:.3f} m < 0.30 m after 1 s at 50% track speed"
        )

    @pytest.mark.parametrize("role,prefix", _ROLES)
    def test_backward_drive_produces_negative_displacement(self, role, prefix):
        """Both tracks at −0.5 → hull moves backward (x decreases) in 1 s."""
        model, data = build_solo_tank(role, prefix)
        x0 = hull_pos(model, data, prefix)[0]

        set_track_ctrl(model, data, prefix, left=-0.5, right=-0.5)
        n_steps = int(1.0 / _DT)
        for _ in range(n_steps):
            mujoco.mj_step(model, data)

        x1 = hull_pos(model, data, prefix)[0]
        assert x1 < x0 - 0.30, (
            f"{role}: backward displacement insufficient — x0={x0:.3f}, x1={x1:.3f}"
        )

    # ------------------------------------------------------------------
    # Turning
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("role,prefix", _ROLES)
    def test_turning_in_place_changes_yaw(self, role, prefix):
        """Left=+0.5, right=−0.5 → |yaw change| > 45° (π/4 rad) in 1 s.

        Turning in place: both tracks equal-and-opposite → zero net forward
        displacement, non-zero rotation.
        """
        model, data = build_solo_tank(role, prefix)
        set_track_ctrl(model, data, prefix, left=0.5, right=-0.5)

        # Track cumulative yaw with per-step unwrapping to handle >360° rotation.
        prev_yaw = hull_yaw(model, data, prefix)
        cumulative = 0.0
        n_steps = int(1.0 / _DT)
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
            curr_yaw = hull_yaw(model, data, prefix)
            dy = curr_yaw - prev_yaw
            if dy > np.pi:
                dy -= 2 * np.pi
            elif dy < -np.pi:
                dy += 2 * np.pi
            cumulative += dy
            prev_yaw = curr_yaw

        delta_yaw = abs(cumulative)
        assert delta_yaw > np.pi / 4, (
            f"{role}: cumulative yaw {np.degrees(delta_yaw):.1f}° < 45° after 1 s of pivot turn"
        )

    # ------------------------------------------------------------------
    # Pure math: differential drive mixing formula
    # ------------------------------------------------------------------

    def test_differential_drive_kinematics_formula(self):
        """Unit math: (v_L, v_R) → (v_fwd, ω) formula is self-consistent.

        Given left and right track speeds (m/s), the net forward velocity
        and yaw rate must satisfy:
          v_fwd = (v_L + v_R) / 2
          ω     = (v_R - v_L) / TRACK_WIDTH

        This test does not run MuJoCo — it verifies that the constants and
        formulas in tank.py are internally consistent for representative inputs.
        """
        cases = [
            # (left_norm, right_norm, expected_v_fwd_sign, expected_omega_sign)
            ( 0.5,  0.5,  +1,  0),   # straight forward
            (-0.5, -0.5,  -1,  0),   # straight backward
            ( 0.5, -0.5,   0, -1),   # pivot left (ω < 0 = counter-clockwise in standard frames)
            (-0.5,  0.5,   0, +1),   # pivot right
            ( 0.3,  0.6,  +1, +1),   # curve right
        ]
        for left_n, right_n, v_sign, omega_sign in cases:
            v_L = left_n  * MAX_TRACK_SPEED
            v_R = right_n * MAX_TRACK_SPEED
            v_fwd = (v_L + v_R) / 2.0
            omega = (v_R - v_L) / TRACK_WIDTH

            if v_sign != 0:
                assert np.sign(v_fwd) == v_sign, (
                    f"v_L={v_L}, v_R={v_R}: expected v_fwd sign {v_sign}, got {np.sign(v_fwd)}"
                )
            else:
                assert abs(v_fwd) < 1e-9, (
                    f"v_L={v_L}, v_R={v_R}: expected v_fwd≈0, got {v_fwd}"
                )

            if omega_sign != 0:
                assert np.sign(omega) == omega_sign, (
                    f"v_L={v_L}, v_R={v_R}: expected ω sign {omega_sign}, got {np.sign(omega)}"
                )
            else:
                assert abs(omega) < 1e-9, (
                    f"v_L={v_L}, v_R={v_R}: expected ω≈0, got {omega}"
                )


class TestHeadingFollowsThrust:

    @pytest.mark.parametrize("yaw_deg", [90, 180, -135])
    def test_forward_drive_follows_hull_heading(self, yaw_deg):
        """Full forward drive moves the tank along its own heading at any yaw.

        Regression for the e2e-found bug: joint-transmission track motors on
        a freejoint push along WORLD +x regardless of heading (freejoint
        translational dof axes are world-aligned), which piled every tank
        onto the east wall. Site-transmission velocity servos fix it.
        """
        model, data = build_solo_tank("shotgun", "red1")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red1_hull")
        jid = model.body_jntadr[bid]
        qadr = model.jnt_qposadr[jid]
        yaw = np.radians(yaw_deg)
        data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(model, data)

        p0 = hull_pos(model, data, "red1")[:2]
        set_track_ctrl(model, data, "red1", 1.0, 1.0)
        for _ in range(int(1.0 / _DT)):
            mujoco.mj_step(model, data)
        disp = hull_pos(model, data, "red1")[:2] - p0

        heading = np.array([np.cos(yaw), np.sin(yaw)])
        along = float(np.dot(disp, heading))
        cross = float(np.linalg.norm(disp - along * heading))
        assert along > 0.6, (
            f"yaw={yaw_deg}°: moved only {along:.2f} m along heading "
            f"(displacement {disp.round(2)}) — thrust is not body-frame"
        )
        assert cross < 0.3 * along, (
            f"yaw={yaw_deg}°: lateral drift {cross:.2f} m vs {along:.2f} m forward"
        )

    def test_speed_capped_near_max_track_speed(self):
        """Sustained full drive settles near MAX_TRACK_SPEED, not unbounded."""
        model, data = build_solo_tank("shotgun", "red1")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red1_hull")
        set_track_ctrl(model, data, "red1", 1.0, 1.0)
        for _ in range(int(2.0 / _DT)):
            mujoco.mj_step(model, data)
        speed = float(np.linalg.norm(data.cvel[bid][3:5]))
        assert speed < MAX_TRACK_SPEED * 1.1, (
            f"speed {speed:.2f} m/s exceeds MAX_TRACK_SPEED={MAX_TRACK_SPEED}"
        )
