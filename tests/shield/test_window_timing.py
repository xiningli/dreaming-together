"""Window open/close timing through real physics (G0, design §3.2).

The elbow was removed in v2; these tests prove the 2-DOF arm can still run
the window mechanic at combat-relevant speed. Unlike the geometry tests in
test_shield.py (which teleport the arm via set_arm_ctrl), these command a
PD target with set_arm_target and step the simulation, measuring the real
settling time of the arm under ARM_KP/ARM_KD.

Acceptance (from the Stage 1 curriculum): open to W_q > 0.5 within 400 ms
of the cue; re-close to W_q < 0.1 within 400 ms of the close cue.

Also here: randomized anti-tunneling trials (rule R5) — pellets fired at
the raised shield from varied offsets must never pass through.

Run: python -m pytest tests/shield/test_window_timing.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.envs.tank import (
    ARM_PAN_RANGE,
    PELLET_SPEED,
    SHIELD_WIDTH,
    SHIELD_HEIGHT,
    window_quality,
    set_arm_ctrl,
    set_arm_target,
)
from tests.helpers import (
    build_interception_range,
    spawn_pellet,
    contacts_between,
    MUZZLE_Z,
)

_DT = 0.002
_CUE_BUDGET_S = 0.400          # max time from cue to reaching the W_q criterion
_SHIELD_DIST  = 1.5
_TARGET_DIST  = 3.0


def _steps(seconds: float) -> int:
    return int(round(seconds / _DT))


class TestWindowTiming:

    def test_window_opens_within_400ms(self):
        """From the raised pose, commanding max pan must reach W_q > 0.5
        within 400 ms of simulated time (HARD open criterion)."""
        model, data = build_interception_range(_SHIELD_DIST, _TARGET_DIST)
        set_arm_ctrl(model, data, "red0", pan_rad=0.0, tilt_rad=0.0)
        mujoco.mj_forward(model, data)
        assert window_quality(model, data, "shooter_muzzle", "red0") < 0.05

        set_arm_target(model, data, "red0", pan_rad=ARM_PAN_RANGE[1], tilt_rad=0.0)
        t_reach = None
        for i in range(_steps(_CUE_BUDGET_S)):
            mujoco.mj_step(model, data)
            if window_quality(model, data, "shooter_muzzle", "red0") > 0.5:
                t_reach = (i + 1) * _DT
                break

        assert t_reach is not None, (
            f"W_q did not exceed 0.5 within {_CUE_BUDGET_S*1000:.0f} ms of the "
            f"open cue (2-DOF arm too slow — check ARM_KP/ARM_KD or pan range)"
        )

    def test_window_recloses_within_400ms(self):
        """From the fully open pose, commanding pan=0 must return to
        W_q < 0.1 within 400 ms (close criterion)."""
        model, data = build_interception_range(_SHIELD_DIST, _TARGET_DIST)
        set_arm_ctrl(model, data, "red0", pan_rad=ARM_PAN_RANGE[1], tilt_rad=0.0)
        mujoco.mj_forward(model, data)
        assert window_quality(model, data, "shooter_muzzle", "red0") > 0.5

        set_arm_target(model, data, "red0", pan_rad=0.0, tilt_rad=0.0)
        t_reach = None
        for i in range(_steps(_CUE_BUDGET_S)):
            mujoco.mj_step(model, data)
            if window_quality(model, data, "shooter_muzzle", "red0") < 0.1:
                t_reach = (i + 1) * _DT
                break

        assert t_reach is not None, (
            f"W_q did not fall below 0.1 within {_CUE_BUDGET_S*1000:.0f} ms of "
            f"the close cue"
        )


class TestAntiTunneling:

    def test_ray_swept_pellets_never_tunnel_through_raised_shield(self):
        """200 randomized pellet trials against the raised shield through the
        ProjectileManager (rule R5): every pellet must register a sweep hit
        on the shield geom, and no pellet may ever be observed behind the
        shield plane while active.

        Contact physics alone let 1/200 oblique pellets through (found by
        this gate in its first run); the ray sweep is the authoritative hit
        detector precisely because it cannot tunnel.
        """
        from dreaming_together.envs.projectiles import ProjectileManager

        rng = np.random.default_rng(7)
        n_trials = 200
        model, data = build_interception_range(_SHIELD_DIST, _TARGET_DIST)
        shield_x = _SHIELD_DIST   # shield face is between muzzle and hull
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pellet_0")

        non_shield_hits = []
        behind = 0
        for trial in range(n_trials):
            mujoco.mj_resetData(model, data)
            set_arm_ctrl(model, data, "red0", pan_rad=0.0, tilt_rad=0.0)
            mujoco.mj_forward(model, data)
            pm = ProjectileManager(model, data)

            # Aim at a random point on the central shield area.
            dy = rng.uniform(-SHIELD_WIDTH * 0.35, SHIELD_WIDTH * 0.35)
            dz = rng.uniform(-0.25, min(0.25, SHIELD_HEIGHT - MUZZLE_Z))
            origin = np.array([0.0, 0.0, MUZZLE_Z])
            aim    = np.array([shield_x, dy, MUZZLE_Z + dz])
            vel    = (aim - origin)
            vel    = vel / np.linalg.norm(vel) * PELLET_SPEED
            pm.spawn(origin, vel, shooter="test")

            for _ in range(150):   # 300 ms
                mujoco.mj_step(model, data)
                pm.step()
                if pm.n_active and data.xpos[bid][0] > shield_x + 0.30:
                    behind += 1
                    break

            hits = pm.drain_hits()
            assert hits, f"trial {trial}: pellet registered no hit at all"
            if hits[0].geom_name != "red0_shield_g":
                non_shield_hits.append((trial, hits[0].geom_name))

        assert behind == 0, (
            f"{behind}/{n_trials} active pellets observed behind the raised "
            f"shield — ray sweep failed to retire them at the shield face"
        )
        assert not non_shield_hits, (
            f"Pellets aimed at the shield centre hit other geoms first: "
            f"{non_shield_hits[:5]}"
        )
