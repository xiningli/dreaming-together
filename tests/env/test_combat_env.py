"""CombatEnv game-layer tests (G2 support).

What these prove
----------------
- Spawn randomization stays in U(1.5, 5.0) and is mirror-symmetric.
- The action interface drives all four agents; obs are 16-dim and finite.
- A scripted point-blank volley reduces the victim's HP, ends the episode,
  and assigns win/loss correctly (one death = team loss).
- Pellet-hit and friendly-fire events land in the right agents' rewards.
- Timeout produces a draw with the draw penalty for all agents.
- Tank-tank collision: two hulls driven into each other collide and do not
  interpenetrate (the v0 "kick" scenario, never before tested).

Run: python -m pytest tests/env/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.envs.combat_env import (
    CombatEnv, PREFIXES, OBS_DIM, HP_MAX,
)
from dreaming_together.envs.rewards import (
    R_PELLET_HIT, R_FRIENDLY_FIRE, R_WIN, R_LOSS, R_DRAW,
)
from dreaming_together.envs.tank import hull_pos, hull_yaw, HULL_LENGTH
from dreaming_together.oracle import ScriptedTeam
from tools.ik_expert import aim_angles
from dreaming_together.envs.tank import ARM_PAN_RANGE, ARM_TILT_RANGE


def _idle():
    return {p: np.array([0, 0, 0, 0, -1.0]) for p in PREFIXES}


def _aim_action(env, shooter, target_prefix, trigger):
    tgt = hull_pos(env.model, env.data, target_prefix) + [0, 0, 0.15]
    pan, tilt = aim_angles(env.model, env.data, shooter, tgt)
    a = np.array([0, 0,
                  2*(pan - ARM_PAN_RANGE[0])/(ARM_PAN_RANGE[1]-ARM_PAN_RANGE[0]) - 1,
                  2*(tilt - ARM_TILT_RANGE[0])/(ARM_TILT_RANGE[1]-ARM_TILT_RANGE[0]) - 1,
                  1.0 if trigger else -1.0])
    return a


class TestSpawns:

    def test_separation_in_range_and_varies(self):
        env = CombatEnv(seed=3)
        seps = []
        for k in range(20):
            env.reset(seed=k)
            seps.append(env.spawn_separation)
            for p in PREFIXES:
                x = hull_pos(env.model, env.data, p)[0]
                assert abs(abs(x) - env.spawn_separation / 2) < 1e-6
        assert all(1.5 <= s <= 5.0 for s in seps)
        assert np.std(seps) > 0.3, "separation not randomized"

    def test_mirror_symmetry(self):
        env = CombatEnv(seed=5)
        env.reset(seed=7)
        for red, blue in (("red0", "blue0"), ("red1", "blue1")):
            pr = hull_pos(env.model, env.data, red)
            pb = hull_pos(env.model, env.data, blue)
            np.testing.assert_allclose(pr[:2], -pb[:2], atol=1e-6)
            yr = hull_yaw(env.model, env.data, red)
            yb = hull_yaw(env.model, env.data, blue)
            assert abs(abs(yr - yb) - np.pi) < 1e-6


class TestStepAPI:

    def test_obs_shape_and_finite(self):
        env = CombatEnv(seed=1)
        obs = env.reset(seed=1)
        assert set(obs) == set(PREFIXES)
        for p in PREFIXES:
            assert obs[p].shape == (OBS_DIM,)
            assert np.all(np.isfinite(obs[p]))
        obs, rewards, done, info = env.step(_idle())
        assert rewards.shape == (4,)
        assert not done

    def test_point_blank_volley_damages_and_ends_episode(self):
        env = CombatEnv(seed=2, spawn_sep_range=(2.0, 2.0))
        env.reset(seed=2)
        hits_reward_seen = False
        for step in range(400):
            actions = _idle()
            actions["red1"] = _aim_action(env, "red1", "blue1",
                                          trigger=step > 2)
            obs, rewards, done, info = env.step(actions)
            if info["pellet_hits"][1] > 0:
                assert rewards[1] > 0, "shooter got no positive reward for hits"
                hits_reward_seen = True
            if done:
                break
        assert done, "point-blank volleys never ended the episode"
        assert hits_reward_seen
        assert env.hp["blue1"] == 0
        assert tuple(env.team_result) == (1, -1), "red should win"
        # terminal rewards: winners +10 among their sum, losers -10
        assert rewards[0] > R_WIN / 2 and rewards[2] < R_LOSS / 2

    def test_friendly_fire_penalized(self):
        env = CombatEnv(seed=4, spawn_sep_range=(3.0, 3.0))
        env.reset(seed=4)
        # red0 sits 90° to red1's left — outside the ±60° pan range — so
        # rotate red1's hull to face its teammate before firing.
        qadr, _ = env._hull_qadr["red1"]
        yaw = np.pi / 2
        env.data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0,
                                            np.sin(yaw / 2)]
        mujoco.mj_forward(env.model, env.data)

        ff_seen = False
        for step in range(200):
            actions = _idle()
            actions["red1"] = _aim_action(env, "red1", "red0",
                                          trigger=step > 2)
            obs, rewards, done, info = env.step(actions)
            i = PREFIXES.index("red1")
            if info["damage"][0] > 0:      # red0 took damage from red1
                assert rewards[i] < 0, (
                    "friendly fire produced non-negative reward for shooter")
                ff_seen = True
                break
            if done:
                break
        assert ff_seen, "friendly-fire scenario never landed a pellet"

    def test_timeout_is_draw_with_penalty(self):
        env = CombatEnv(seed=6, episode_cap_s=0.5)
        env.reset(seed=6)
        done = False
        while not done:
            obs, rewards, done, info = env.step(_idle())
        assert tuple(env.team_result) == (0, 0)
        assert np.all(rewards < R_DRAW / 2), (
            f"draw penalty missing from terminal rewards: {rewards}")

    def test_oracle_episode_terminates(self):
        env = CombatEnv(seed=8)
        env.reset(seed=8)
        teams = (ScriptedTeam(0), ScriptedTeam(1))
        done = False
        steps = 0
        while not done and steps < 700:
            actions = {}
            for tm in teams:
                actions.update(tm.act(env))
            obs, rewards, done, info = env.step(actions)
            steps += 1
        assert done


class TestTankCollision:

    def test_hulls_collide_and_do_not_interpenetrate(self):
        """Two tanks driven head-on into each other must make hull contact
        and never overlap (v0 'kick' scenario — physical robustness)."""
        # red1 (-sep/2, -0.6, yaw 0) and blue0 (+sep/2, -0.6, yaw pi) share
        # the same y-lane, so full forward drive is a true head-on approach.
        env = CombatEnv(seed=9, spawn_sep_range=(2.0, 2.0))
        env.reset(seed=9)
        min_gap = np.inf
        contact_seen = False
        ga = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "red1_hull_g")
        gb = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "blue0_hull_g")
        for _ in range(80):   # 4 s of full-speed head-on driving
            actions = _idle()
            actions["red1"] = np.array([1, 1, 0, 0, -1.0])
            actions["blue0"] = np.array([1, 1, 0, 0, -1.0])
            obs, rewards, done, info = env.step(actions)
            for c in range(env.data.ncon):
                con = env.data.contact[c]
                if {con.geom1, con.geom2} == {ga, gb}:
                    contact_seen = True
            d = np.linalg.norm(
                hull_pos(env.model, env.data, "red1")[:2]
                - hull_pos(env.model, env.data, "blue0")[:2])
            min_gap = min(min_gap, d)
            for p in ("red1", "blue0"):
                z = hull_pos(env.model, env.data, p)[2]
                assert z < 0.4, f"{p} hull climbed to z={z:.2f} in collision"
        assert contact_seen, "hulls never made contact driving head-on"
        assert min_gap > HULL_LENGTH * 0.9, (
            f"hull centres reached {min_gap:.2f} m — interpenetration "
            f"(hull length {HULL_LENGTH} m)")
