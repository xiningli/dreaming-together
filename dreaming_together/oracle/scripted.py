"""Scripted team oracle — curriculum opponent and G2 gate driver.

Generalized from the choreography that passed the e2e visual test:
  - SHOTGUN: advance toward the nearest living enemy, halt in range, track
    it with the closed-form IK expert, fire on cooldown when the lane past
    the own shield is clear (W_q ≥ 0.5).
  - SHIELD: hold a screening position on the line from the own shotgun to
    the nearest enemy, offset laterally so the shotgun's open-window lane
    clears the shield agent's own hull (design note 13); keep the plate
    raised, and open it only in the beat before the shotgun is ready —
    open-fire-close, the pattern the learned coordinator must eventually
    produce.

Oracles read privileged env state but act ONLY through the same
[-1,1]^5 action interface as learned policies (integrity: one action
pathway).
"""
from __future__ import annotations

import numpy as np

from dreaming_together.envs.combat_env import (
    CombatEnv, PREFIXES, TEAM_OF, SHOTGUN_OF_TEAM, SHIELD_OF_TEAM,
    SHOTGUN_RANGE,
)
from dreaming_together.envs.tank import (
    ARM_PAN_RANGE, ARM_TILT_RANGE, hull_pos, hull_yaw,
)
from tools.ik_expert import aim_angles


def _norm_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _norm(v, lo, hi):
    return float(np.clip(2.0 * (v - lo) / (hi - lo) - 1.0, -1.0, 1.0))


class ScriptedTeam:
    """Produces actions for one team's shield + shotgun pair."""

    ENGAGE_DIST = 2.4          # shotgun halts here
    SCREEN_AHEAD = 1.1         # shield stands this far ahead of the shotgun
    SCREEN_LATERAL = 0.5       # ... offset out of the shotgun's firing lane
    WINDOW_LEAD_S = 0.35       # open the window this long before ready
    FIRE_WQ = 0.5

    def __init__(self, team: int):
        self.team = team
        self.shotgun = SHOTGUN_OF_TEAM[team]
        self.shield = SHIELD_OF_TEAM[team]

    # -- low-level drive: returns (left, right) in [-1, 1] ---------------
    def _drive(self, env: CombatEnv, p: str, target_xy: np.ndarray,
               halt_dist: float, face_xy: np.ndarray | None = None):
        pos = hull_pos(env.model, env.data, p)[:2]
        yaw = hull_yaw(env.model, env.data, p)
        v = target_xy - pos
        dist = float(np.linalg.norm(v))
        if dist < halt_dist:
            if face_xy is None:
                return 0.0, 0.0
            want = np.arctan2(*(face_xy - pos)[::-1])
            err = _norm_angle(want - yaw)
            turn = float(np.clip(1.2 * err, -0.4, 0.4))
            return -turn, turn
        bearing = _norm_angle(np.arctan2(v[1], v[0]) - yaw)
        fwd = 0.6 * np.cos(bearing)
        turn = float(np.clip(1.5 * bearing, -0.5, 0.5))
        return fwd - turn, fwd + turn

    def _nearest_enemy(self, env: CombatEnv, p: str) -> str:
        opps = [o for o in PREFIXES if TEAM_OF[o] != self.team]
        alive = [o for o in opps if env.hp[o] > 0] or opps
        pos = hull_pos(env.model, env.data, p)[:2]
        return min(alive, key=lambda o: np.linalg.norm(
            hull_pos(env.model, env.data, o)[:2] - pos))

    # ---------------------------------------------------------------------
    def act(self, env: CombatEnv) -> dict[str, np.ndarray]:
        actions = {}
        sg, sh = self.shotgun, self.shield
        sg_pos = hull_pos(env.model, env.data, sg)[:2]
        enemy = self._nearest_enemy(env, sg)
        enemy_pos3 = hull_pos(env.model, env.data, enemy)
        enemy_pos = enemy_pos3[:2]
        dist_to_enemy = float(np.linalg.norm(enemy_pos - sg_pos))

        # SHOTGUN ---------------------------------------------------------
        left, right = self._drive(env, sg, enemy_pos, self.ENGAGE_DIST,
                                  face_xy=enemy_pos)
        pan, tilt = aim_angles(env.model, env.data, sg,
                               enemy_pos3 + np.array([0.0, 0.0, 0.15]))
        ready = env.cooldown[sg] <= 0.0
        in_range = dist_to_enemy < SHOTGUN_RANGE
        clear = env._team_wq(self.team) >= self.FIRE_WQ
        trigger = 1.0 if (ready and in_range and clear) else -1.0
        actions[sg] = np.array([left, right,
                                _norm(pan, *ARM_PAN_RANGE),
                                _norm(tilt, *ARM_TILT_RANGE),
                                trigger])

        # SHIELD ----------------------------------------------------------
        lane = enemy_pos - sg_pos
        lane_n = lane / (np.linalg.norm(lane) + 1e-9)
        perp = np.array([-lane_n[1], lane_n[0]])
        screen = sg_pos + lane_n * self.SCREEN_AHEAD + perp * self.SCREEN_LATERAL
        left, right = self._drive(env, sh, screen, 0.2, face_xy=enemy_pos)
        window_open = (env.cooldown[sg] <= self.WINDOW_LEAD_S
                       and in_range and env.hp[sg] > 0)
        pan_target = ARM_PAN_RANGE[1] if window_open else 0.0
        actions[sh] = np.array([left, right,
                                _norm(pan_target, *ARM_PAN_RANGE),
                                _norm(0.0, *ARM_TILT_RANGE),
                                -1.0])
        return actions


class EliteScriptedTeam(ScriptedTeam):
    """Calibrated evaluation opponent — FROZEN 2026-07-04.

    G6's causal-necessity ablation under-measures against the standard
    oracle: by Phase C the learned team beats it nearly deaf (drop 4 pts).
    Against this tightened variant the same checkpoints show
    win(z_g on)=0.95 / win(z_g off)=0.75 — a 20.7-pt communication margin.
    Calibrated once on condition C; used identically for every condition
    in G6 and all A/B/C evaluations (integrity: no per-condition opponent
    tuning).
    """
    ENGAGE_DIST = 1.8
    WINDOW_LEAD_S = 0.20
    FIRE_WQ = 0.30
    SCREEN_LATERAL = 0.40
