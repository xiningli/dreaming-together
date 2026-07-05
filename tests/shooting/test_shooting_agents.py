"""Shooting on real agent bodies: friendly and hostile targets.

These tests verify physical pellet contact against actual tank bodies
(not a static paper target). They cover:

  Hostile shotgun hull  — pellet reaches a blue team shotgun
  Hostile shield body   — pellet reaches a blue team shield agent's hull when
                          the shield arm is rotated out of the corridor
  Hostile shield block  — pellet is stopped by the blue team's shield geom
                          when the arm is in the blocking pose
  Friendly fire         — pellet contacts a red teammate's hull
  Own shield block      — friendly shield in column formation stops own pellet
                          before it reaches the enemy

Minimum models
--------------
  Hostile tests:   build_opposing_pair(red1, shotgun, blue1/blue0, ...)
  Friendly fire:   build_friendly_pair()
  Column block:    build_column_with_enemy()

What these tests do NOT prove
------------------------------
  - HP deduction (game-layer, not physics)
  - Reward attribution (covered in tests/reward/)
  - Multi-pellet cone spread (covered in test_shooting.py)
  - Shield blocking when agents are moving (full combat env, not physics unit)

Run: python -m pytest tests/shooting/test_shooting_agents.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.envs.tank import (
    PELLET_SPEED,
    ARM_PAN_RANGE,
    set_arm_ctrl,
    muzzle_pos,
    muzzle_dir,
)
from tests.helpers import (
    build_opposing_pair,
    build_friendly_pair,
    build_column_with_enemy,
    geom_id,
    any_contact_with,
    contacts_between,
    spawn_pellet,
    MUZZLE_Z,
)

_ARM_PAN_RAISED  = 0.0
_ARM_PAN_TWISTED = ARM_PAN_RANGE[1]     # 60 deg — maximum pan, opens the window


def _fire_pellet_from(model, data, shooter_prefix: str) -> None:
    """Spawn pellet_0 at shooter's muzzle pointing in muzzle direction."""
    pos = muzzle_pos(model, data, shooter_prefix)
    vel = muzzle_dir(model, data, shooter_prefix) * PELLET_SPEED
    spawn_pellet(model, data, "pellet_0", pos, vel)


class TestShootingAgents:

    def test_pellet_hits_hostile_shotgun_hull(self):
        """red1 fires toward blue1 (hostile shotgun, no shield): pellet contacts blue1_hull_g.

        The simplest hostile-fire scenario.  No shield agent is present.
        Tests that a pellet aimed at a hostile body actually makes contact.
        """
        model, data = build_opposing_pair(
            shooter_prefix="red1", shooter_role="shotgun",
            target_prefix="blue1", target_role="shotgun",
        )
        mujoco.mj_forward(model, data)
        _fire_pellet_from(model, data, "red1")

        hit = False
        for _ in range(150):   # 300 ms at dt=0.002
            mujoco.mj_step(model, data)
            if any_contact_with(data, model, "blue1_hull_g"):
                hit = True
                break

        assert hit, (
            "Pellet did not contact blue1_hull_g within 300 ms. "
            "Check muzzle FK (muzzle_pos, muzzle_dir) and contype/conaffinity "
            "on hull geom and pellet body."
        )

    def test_pellet_hits_hostile_shield_agent_body_when_unshielded(self):
        """red1 fires at blue0 (hostile shield agent) whose arm is twisted away.

        When the shield arm is at maximum pan (60°), the shield face is rotated
        out of the firing corridor.  The pellet must reach blue0's hull body,
        not the shield geom.
        """
        model, data = build_opposing_pair(
            shooter_prefix="red1", shooter_role="shotgun",
            target_prefix="blue0", target_role="shield",
        )
        # Rotate shield arm out of the way — window is fully open
        set_arm_ctrl(model, data, "blue0",
                     pan_rad=_ARM_PAN_TWISTED, tilt_rad=0.0)
        mujoco.mj_forward(model, data)
        _fire_pellet_from(model, data, "red1")

        hull_hit   = False
        shield_hit = False
        for _ in range(150):
            mujoco.mj_step(model, data)
            if any_contact_with(data, model, "blue0_hull_g"):
                hull_hit = True
                break
            if any_contact_with(data, model, "blue0_shield_g"):
                shield_hit = True

        assert hull_hit, (
            f"Pellet did not reach blue0_hull_g (shield_hit={shield_hit}). "
            f"arm_pan={_ARM_PAN_TWISTED:.3f} rad should rotate shield out of corridor."
        )

    def test_pellet_blocked_by_hostile_shield_when_raised(self):
        """red1 fires at blue0 (hostile shield agent) whose arm is raised (pan=0).

        The shield face is perpendicular to the firing axis.  Pellet must strike
        blue0_shield_g and must NOT reach blue0_hull_g.
        """
        model, data = build_opposing_pair(
            shooter_prefix="red1", shooter_role="shotgun",
            target_prefix="blue0", target_role="shield",
        )
        # Shield raised — arm pointing directly toward shooter
        set_arm_ctrl(model, data, "blue0",
                     pan_rad=_ARM_PAN_RAISED, tilt_rad=0.0)
        mujoco.mj_forward(model, data)
        _fire_pellet_from(model, data, "red1")

        shield_hit = False
        hull_hit   = False
        for _ in range(150):
            mujoco.mj_step(model, data)
            if any_contact_with(data, model, "blue0_shield_g"):
                shield_hit = True
            if contacts_between(data, model, "pellet_0_g", "blue0_hull_g"):
                hull_hit = True
                break

        assert shield_hit, (
            "Pellet did not contact blue0_shield_g with shield raised (arm_pan=0). "
            "Check shield geom is in the line of fire and contype/conaffinity allows contact."
        )
        assert not hull_hit, (
            "Pellet reached blue0_hull_g despite raised shield — "
            "shield is not physically blocking the projectile."
        )

    def test_friendly_fire_hits_teammate_hull(self):
        """red1 fires toward red0 (same team, shield agent): pellet contacts red0_hull_g.

        Friendly fire is not physically prevented by the environment.  The game
        layer must detect it through contact and apply the -0.3 per pellet penalty.
        This test verifies the contact is detectable.
        """
        model, data = build_friendly_pair(distance=2.5)
        mujoco.mj_forward(model, data)
        _fire_pellet_from(model, data, "red1")

        hit = False
        for _ in range(150):
            mujoco.mj_step(model, data)
            if any_contact_with(data, model, "red0_hull_g"):
                hit = True
                break

        assert hit, (
            "Pellet from red1 did not contact red0_hull_g (teammate). "
            "Friendly-fire contact must be detectable so the game layer "
            "can apply the R_FRIENDLY_FIRE penalty."
        )

    def test_own_shield_blocks_own_shot_in_column(self):
        """red0 (friendly shield, arm raised) stops red1's pellet in column formation.

        Column layout:
          x=0.0  red1 (shotgun)
          x=1.5  red0 (shield, arm_pan=0 → shield faces +x, blocking the corridor)
          x=3.0  blue1 (enemy)

        With the window closed (arm not rotated), red1 cannot shoot through to
        blue1 — the friendly shield geom intercepts the pellet.
        This is the defining physics constraint of the window mechanic:
        the coordinator must open the window (rotate red0's shield) before
        red1 can fire effectively.
        """
        model, data = build_column_with_enemy(shield_distance=1.5, enemy_distance=3.0)
        # Shield raised — corridor blocked
        set_arm_ctrl(model, data, "red0",
                     pan_rad=_ARM_PAN_RAISED, tilt_rad=0.0)
        mujoco.mj_forward(model, data)
        _fire_pellet_from(model, data, "red1")

        own_shield_hit = False
        enemy_hit      = False
        for _ in range(200):   # 400 ms — generous budget so enemy would clearly be hit if unblocked
            mujoco.mj_step(model, data)
            if any_contact_with(data, model, "red0_shield_g"):
                own_shield_hit = True
            if contacts_between(data, model, "pellet_0_g", "blue1_hull_g"):
                enemy_hit = True
                break

        assert own_shield_hit, (
            "Pellet did not hit the friendly shield (red0_shield_g) in column formation. "
            "Own shield must physically block the firing corridor when window is closed."
        )
        assert not enemy_hit, (
            "Pellet reached blue1_hull_g despite friendly shield in the way. "
            "Window mechanic requires the friendly shield to block own shots."
        )
