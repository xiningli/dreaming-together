"""Shield interception tests.

Minimum model: fixed world-frame muzzle origin + shield tank (red0) + paper target.

Layout:
  x=0.0  world site "shooter_muzzle"   (no hull body — just a ray origin)
  x=1.5  shield tank (red0), arm centred at x=1.5
  x=3.0  paper target

Why no shooter hull?
  The shooter body is irrelevant to shield geometry.  A world site is
  sufficient for mj_ray calls and pellet spawning, and its absence avoids
  confounding failures (shooter arm FK errors, etc.).

What these tests prove
----------------------
- Shield geom exists with correct dimensions (1.2 m × 1.0 m × 0.04 m).
- With shield raised (arm at rest pose), a ray from the muzzle origin to the
  target is intercepted by the shield geom.
- With shield twisted (arm_pan rotated), the same ray passes through to the
  target (window is open).
- W_q = 0 when shield fully blocks the corridor; W_q > 0 when twisted.
- C_r ≈ 1 when shield is raised; C_r ≈ 0 when shield is perpendicular.
- A physical pellet spawned at the muzzle origin is stopped by the raised
  shield and does NOT reach the paper target.

What these tests do NOT prove
------------------------------
- Window quality under moving agents (requires full combat env).
- Friendly-fire accounting (covered in env tests).
- Shield durability or HP impact (shield blocks pellets with no HP cost by
  design — the test confirms pellets stop, not that they deal damage to the
  shield).

Run: python -m pytest tests/shield/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.envs.tank import (
    SHIELD_WIDTH,
    SHIELD_HEIGHT,
    SHIELD_THICKNESS,
    PELLET_SPEED,
    window_quality,
    residual_coverage,
    set_arm_ctrl,
)
from tests.helpers import (
    build_interception_range,
    build_solo_tank,
    cast_ray,
    any_contact_with,
    spawn_pellet,
    geom_id,
    site_id,
    MUZZLE_Z,
)

_SHIELD_DIST  = 1.5
_TARGET_DIST  = 3.0
_DT = 0.002

# arm_pan angle (rad) considered "fully raised" (arm forward, shield face ⊥ to x-axis)
_ARM_PAN_RAISED  = 0.0
# arm_pan angle that opens the window (shield rotated ~60° off frontal axis)
_ARM_PAN_TWISTED = 1.047    # 60° — equal to ARM_PAN_RANGE max


def _muzzle_origin(model: mujoco.MjModel,
                   data: mujoco.MjData) -> np.ndarray:
    """World position of the shooter_muzzle site."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "shooter_muzzle")
    return data.site_xpos[sid].copy()


def _target_centre(dist: float = _TARGET_DIST) -> np.ndarray:
    return np.array([dist, 0.0, MUZZLE_Z])


class TestShield:

    # ------------------------------------------------------------------
    # Shield geometry
    # ------------------------------------------------------------------

    def test_shield_geom_exists(self):
        """Model contains a geom named 'red0_shield_g' with positive id."""
        model, data = build_solo_tank("shield", "red0")
        gid = geom_id(model, "red0_shield_g")
        assert gid >= 0, "Geom 'red0_shield_g' not found in solo shield model"

    def test_shield_geom_dimensions(self):
        """Shield geom half-sizes match spec: (THICKNESS/2, WIDTH/2, HEIGHT/2).

        MuJoCo stores box half-sizes in geom_size[gid, 0:3].
        """
        model, data = build_solo_tank("shield", "red0")
        gid = geom_id(model, "red0_shield_g")
        half_sizes = model.geom_size[gid][:3]

        expected = np.array([
            SHIELD_THICKNESS / 2,   # depth (x — thin dimension)
            SHIELD_WIDTH     / 2,   # width (y)
            SHIELD_HEIGHT    / 2,   # height (z)
        ])
        np.testing.assert_allclose(
            half_sizes, expected, atol=1e-4,
            err_msg=(
                f"Shield half-sizes {half_sizes} do not match spec {expected}. "
                f"Expected (depth={SHIELD_THICKNESS/2:.3f}, "
                f"width={SHIELD_WIDTH/2:.3f}, height={SHIELD_HEIGHT/2:.3f})"
            )
        )

    # ------------------------------------------------------------------
    # Ray interception
    # ------------------------------------------------------------------

    def test_raised_shield_intercepts_ray(self):
        """With arm_pan=0 (shield facing shooter), ray hits shield, not target.

        A ray cast from shooter_muzzle toward the paper target must strike
        the shield geom (red0_shield_g) before reaching paper_target_g.
        """
        model, data = build_interception_range(_SHIELD_DIST, _TARGET_DIST)
        set_arm_ctrl(model, data, "red0",
                     pan_rad=_ARM_PAN_RAISED, tilt_rad=0.0)
        mujoco.mj_forward(model, data)

        origin    = _muzzle_origin(model, data)
        direction = _target_centre() - origin
        _, hit_geom = cast_ray(model, data, origin, direction)

        shield_gid = geom_id(model, "red0_shield_g")
        target_gid = geom_id(model, "paper_target_g")
        assert hit_geom == shield_gid, (
            f"Expected ray to hit shield (id={shield_gid}) but hit geom id={hit_geom} "
            f"(target_g id={target_gid}). "
            f"Shield may not be in the line of fire at arm_pan={_ARM_PAN_RAISED} rad."
        )

    def test_twisted_shield_passes_ray_to_target(self):
        """With arm_pan=ARM_PAN_RANGE_MAX (shield rotated), ray reaches target.

        When the arm is panned to maximum rotation the shield face is
        no longer perpendicular to the shooter→target line; the ray must
        pass through to paper_target_g.
        """
        model, data = build_interception_range(_SHIELD_DIST, _TARGET_DIST)
        set_arm_ctrl(model, data, "red0",
                     pan_rad=_ARM_PAN_TWISTED, tilt_rad=0.0)
        mujoco.mj_forward(model, data)

        origin    = _muzzle_origin(model, data)
        direction = _target_centre() - origin
        _, hit_geom = cast_ray(model, data, origin, direction)

        target_gid = geom_id(model, "paper_target_g")
        assert hit_geom == target_gid, (
            f"Expected ray to pass through to target (id={target_gid}) "
            f"but hit geom id={hit_geom} at arm_pan={_ARM_PAN_TWISTED:.3f} rad."
        )

    # ------------------------------------------------------------------
    # Window quality W_q
    # ------------------------------------------------------------------

    def test_window_quality_zero_when_raised(self):
        """W_q ≈ 0 when shield fully blocks the corridor (arm_pan = 0)."""
        model, data = build_interception_range(_SHIELD_DIST, _TARGET_DIST)
        set_arm_ctrl(model, data, "red0",
                     pan_rad=_ARM_PAN_RAISED, tilt_rad=0.0)
        mujoco.mj_forward(model, data)

        wq = window_quality(model, data,
                            shooter_prefix="shooter_muzzle",
                            shield_prefix="red0")
        assert wq < 0.05, (
            f"W_q={wq:.3f} with raised shield — expected < 0.05 (full block)"
        )

    def test_window_quality_nonzero_when_twisted(self):
        """W_q > 0.25 when shield arm is at maximum pan (window is open)."""
        model, data = build_interception_range(_SHIELD_DIST, _TARGET_DIST)
        set_arm_ctrl(model, data, "red0",
                     pan_rad=_ARM_PAN_TWISTED, tilt_rad=0.0)
        mujoco.mj_forward(model, data)

        wq = window_quality(model, data,
                            shooter_prefix="shooter_muzzle",
                            shield_prefix="red0")
        assert wq > 0.25, (
            f"W_q={wq:.3f} with twisted shield — expected > 0.25 (window open)"
        )

    # ------------------------------------------------------------------
    # Residual coverage C_r
    # ------------------------------------------------------------------

    def test_residual_coverage_when_raised(self):
        """C_r > 0.8 when shield is raised (most rays from attacker are blocked)."""
        model, data = build_interception_range(_SHIELD_DIST, _TARGET_DIST)
        set_arm_ctrl(model, data, "red0",
                     pan_rad=_ARM_PAN_RAISED, tilt_rad=0.0)
        mujoco.mj_forward(model, data)

        cr = residual_coverage(model, data,
                               attacker_prefix="shooter_muzzle",
                               shield_prefix="red0")
        assert cr > 0.80, (
            f"C_r={cr:.3f} with raised shield — expected > 0.80"
        )

    # ------------------------------------------------------------------
    # Physical pellet interception
    # ------------------------------------------------------------------

    def test_raised_shield_stops_pellet_before_target(self):
        """A physical pellet launched from shooter_muzzle hits the shield, not target.

        Spawns a pellet at the muzzle origin with velocity toward the target.
        Steps the simulation for 200 ms.  The paper target must NOT have been
        contacted; the shield geom MUST have been contacted.

        This test complements the ray tests: it verifies the contype/conaffinity
        of the shield geom actually stop projectile bodies, not just mj_ray.
        """
        model, data = build_interception_range(_SHIELD_DIST, _TARGET_DIST)
        set_arm_ctrl(model, data, "red0",
                     pan_rad=_ARM_PAN_RAISED, tilt_rad=0.0)
        mujoco.mj_forward(model, data)

        origin    = _muzzle_origin(model, data)
        direction = (_target_centre() - origin)
        direction = direction / np.linalg.norm(direction)
        vel = direction * PELLET_SPEED
        spawn_pellet(model, data, "pellet_0", origin, vel)

        shield_hit = False
        target_hit = False
        for _ in range(100):   # 200 ms
            mujoco.mj_step(model, data)
            if any_contact_with(data, model, "red0_shield_g"):
                shield_hit = True
            if any_contact_with(data, model, "paper_target_g"):
                target_hit = True
                break

        assert shield_hit, (
            "Pellet did not contact shield (red0_shield_g) — "
            "check contype/conaffinity on shield geom and pellet body"
        )
        assert not target_hit, (
            "Pellet reached paper_target_g despite raised shield — "
            "shield is not physically blocking the projectile"
        )
