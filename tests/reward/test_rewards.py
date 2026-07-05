"""Unit tests for per-agent reward component functions.

No MuJoCo required. Each test exercises one reward function with a
minimal RewardContext and checks the numerical result exactly.

Test naming convention
----------------------
  test_<component>_<scenario>

  <component> names map directly to functions in rewards.py.
  Each test also documents the expected formula so failures pinpoint
  wrong constants vs wrong formula vs wrong indexing.

Run: python -m pytest tests/reward/ -v
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.envs.rewards import (
    RewardContext,
    # individual component functions
    time_alive_reward,
    damage_taken_penalty,
    advance_reward,
    action_rate_penalty,
    result_reward,
    shield_blocking_reward,
    window_timing_reward,
    assist_reward,
    pellet_hit_reward,
    kill_reward,
    close_range_reward,
    shot_through_window_reward,
    friendly_fire_penalty,
    die_first_penalty,
    # constants used in assertions
    R_TIME_ALIVE,
    R_DAMAGE_TAKEN,
    R_ADVANCE,
    R_WIN,
    R_LOSS,
    R_DRAW,
    R_SHIELD_BLOCKS,
    R_WINDOW_GOOD,
    R_WINDOW_BAD,
    R_ASSIST,
    R_PELLET_HIT,
    R_KILL,
    R_CLOSE_RANGE,
    R_WINDOW_4PLUS,
    R_FRIENDLY_FIRE,
    R_DIE_FIRST,
    LAM_AR,
)


# ---------------------------------------------------------------------------
# Shared context factory
# ---------------------------------------------------------------------------

def _base_ctx(**overrides) -> RewardContext:
    """Neutral context: no rewards, no penalties, episode ongoing.

    All distances constant (no advance), no damage, same action repeated
    (no action-rate penalty), all agents alive.
    Caller overrides specific fields to isolate the component under test.
    """
    ctx = RewardContext(
        dt=0.05,
        alive=np.ones(4, dtype=bool),
        damage_taken=np.zeros(4),
        killed_this_step=np.zeros(4, dtype=bool),
        dist_to_nearest_opp=np.full(4, 5.0),
        prev_dist_to_nearest_opp=np.full(4, 5.0),
        action=np.zeros((4, 5)),
        prev_action=np.zeros((4, 5)),
        episode_done=False,
        team_result=np.zeros(2, dtype=int),
        shield_blocks_line=np.zeros(2, dtype=bool),
        window_open=np.zeros(2, dtype=bool),
        opp_in_reload=np.zeros(2, dtype=bool),
        window_assist=np.zeros(2, dtype=bool),
        pellet_hits_opp=np.zeros(4, dtype=int),
        friendly_pellets=np.zeros(4, dtype=int),
        kills=np.zeros(4, dtype=int),
        within_3m=np.zeros(4, dtype=bool),
        shot_through_window_4plus=np.zeros(2, dtype=bool),
        died_before_teammate=np.zeros(4, dtype=bool),
    )
    return replace(ctx, **overrides)


# ---------------------------------------------------------------------------
# time_alive_reward
# ---------------------------------------------------------------------------

class TestTimeAliveReward:

    def test_all_alive_each_gets_time_alive(self):
        """All 4 agents alive → each gets R_TIME_ALIVE per step."""
        r = time_alive_reward(_base_ctx())
        np.testing.assert_allclose(r, np.full(4, R_TIME_ALIVE))

    def test_dead_agent_gets_zero(self):
        """Dead agent (alive[2]=False) must receive 0, not R_TIME_ALIVE."""
        alive = np.ones(4, dtype=bool)
        alive[2] = False
        r = time_alive_reward(_base_ctx(alive=alive))
        assert r[2] == 0.0, f"Dead agent got {r[2]}, expected 0"
        assert r[0] == R_TIME_ALIVE, f"Living agent got {r[0]}, expected {R_TIME_ALIVE}"


# ---------------------------------------------------------------------------
# damage_taken_penalty
# ---------------------------------------------------------------------------

class TestDamageTakenPenalty:

    def test_damage_produces_negative_reward(self):
        """red1 (index 1) took 10 HP → reward = R_DAMAGE_TAKEN * 10 = -0.20."""
        dmg = np.zeros(4)
        dmg[1] = 10.0
        r = damage_taken_penalty(_base_ctx(damage_taken=dmg))
        expected = R_DAMAGE_TAKEN * 10.0   # -0.02 * 10 = -0.20
        assert abs(r[1] - expected) < 1e-9, f"Got {r[1]}, expected {expected}"

    def test_no_damage_no_penalty(self):
        """Zero damage → zero penalty for all agents."""
        r = damage_taken_penalty(_base_ctx())
        np.testing.assert_allclose(r, np.zeros(4))


# ---------------------------------------------------------------------------
# advance_reward
# ---------------------------------------------------------------------------

class TestAdvanceReward:

    def test_closing_distance_gives_positive_reward(self):
        """Distance dropped 0.10 m → advance reward = R_ADVANCE * 0.10."""
        ctx = _base_ctx(
            dist_to_nearest_opp=np.full(4, 4.9),
            prev_dist_to_nearest_opp=np.full(4, 5.0),
        )
        r = advance_reward(ctx)
        expected = R_ADVANCE * 0.10
        np.testing.assert_allclose(r, np.full(4, expected), atol=1e-9)

    def test_moving_away_gives_zero_reward(self):
        """Distance increased → advance reward is clamped to zero, not negative."""
        ctx = _base_ctx(
            dist_to_nearest_opp=np.full(4, 5.1),
            prev_dist_to_nearest_opp=np.full(4, 5.0),
        )
        r = advance_reward(ctx)
        np.testing.assert_allclose(r, np.zeros(4))

    def test_stationary_gives_zero_reward(self):
        """Constant distance → zero advance reward."""
        r = advance_reward(_base_ctx())
        np.testing.assert_allclose(r, np.zeros(4))


# ---------------------------------------------------------------------------
# action_rate_penalty
# ---------------------------------------------------------------------------

class TestActionRatePenalty:

    def test_action_change_gives_negative_penalty(self):
        """Action changed by 1.0 in one dimension → penalty = -LAM_AR * 1^2 * n_changed."""
        prev = np.zeros((4, 5))
        curr = np.zeros((4, 5))
        curr[0, 0] = 1.0   # only red0 changed action 0 by 1.0
        r = action_rate_penalty(_base_ctx(action=curr, prev_action=prev))
        expected = -LAM_AR * (1.0 ** 2)
        assert abs(r[0] - expected) < 1e-9, f"red0 penalty: got {r[0]}, expected {expected}"
        assert r[1] == 0.0, f"red1 should have zero penalty, got {r[1]}"

    def test_same_action_no_penalty(self):
        """Repeated action → zero action-rate penalty."""
        r = action_rate_penalty(_base_ctx())
        np.testing.assert_allclose(r, np.zeros(4))


# ---------------------------------------------------------------------------
# result_reward
# ---------------------------------------------------------------------------

class TestResultReward:

    def test_win_reward_applies_to_winning_team(self):
        """Red team wins → red agents (0, 1) get R_WIN; blue agents (2, 3) get R_LOSS."""
        ctx = _base_ctx(
            episode_done=True,
            team_result=np.array([1, -1]),   # red=+1 win, blue=-1 loss
        )
        r = result_reward(ctx)
        assert abs(r[0] - R_WIN)  < 1e-9, f"red0 expected {R_WIN}, got {r[0]}"
        assert abs(r[1] - R_WIN)  < 1e-9, f"red1 expected {R_WIN}, got {r[1]}"
        assert abs(r[2] - R_LOSS) < 1e-9, f"blue0 expected {R_LOSS}, got {r[2]}"
        assert abs(r[3] - R_LOSS) < 1e-9, f"blue1 expected {R_LOSS}, got {r[3]}"

    def test_draw_applies_to_all_agents(self):
        """Draw → all agents get R_DRAW."""
        ctx = _base_ctx(
            episode_done=True,
            team_result=np.array([0, 0]),
        )
        r = result_reward(ctx)
        np.testing.assert_allclose(r, np.full(4, R_DRAW))

    def test_no_result_reward_during_episode(self):
        """episode_done=False → no result reward regardless of team_result."""
        ctx = _base_ctx(episode_done=False, team_result=np.array([1, -1]))
        r = result_reward(ctx)
        np.testing.assert_allclose(r, np.zeros(4))


# ---------------------------------------------------------------------------
# shield_blocking_reward
# ---------------------------------------------------------------------------

class TestShieldBlockingReward:

    def test_blocking_gives_reward_to_shield_agent(self):
        """shield_blocks_line[0]=True → red0 (agent 0) gets R_SHIELD_BLOCKS."""
        blocks = np.array([True, False])
        r = shield_blocking_reward(_base_ctx(shield_blocks_line=blocks))
        assert abs(r[0] - R_SHIELD_BLOCKS) < 1e-9, (
            f"red0 expected {R_SHIELD_BLOCKS}, got {r[0]}"
        )
        assert r[2] == 0.0, f"blue0 expected 0 (not blocking), got {r[2]}"

    def test_not_blocking_gives_no_reward(self):
        """No team blocking → all shield agents get 0."""
        r = shield_blocking_reward(_base_ctx())
        assert r[0] == 0.0
        assert r[2] == 0.0


# ---------------------------------------------------------------------------
# window_timing_reward
# ---------------------------------------------------------------------------

class TestWindowTimingReward:

    def test_window_open_during_reload_gives_positive(self):
        """window_open[0]=True AND opp_in_reload[0]=True → red0 gets R_WINDOW_GOOD."""
        ctx = _base_ctx(
            window_open=np.array([True, False]),
            opp_in_reload=np.array([True, False]),
        )
        r = window_timing_reward(ctx)
        assert abs(r[0] - R_WINDOW_GOOD) < 1e-9, (
            f"red0 expected {R_WINDOW_GOOD}, got {r[0]}"
        )

    def test_window_open_while_opponent_ready_gives_penalty(self):
        """window_open[0]=True AND opp_in_reload[0]=False → red0 gets R_WINDOW_BAD."""
        ctx = _base_ctx(
            window_open=np.array([True, False]),
            opp_in_reload=np.array([False, False]),
        )
        r = window_timing_reward(ctx)
        assert abs(r[0] - R_WINDOW_BAD) < 1e-9, (
            f"red0 expected {R_WINDOW_BAD}, got {r[0]}"
        )

    def test_window_closed_gives_zero(self):
        """window_open=[False,False] → zero for all shield agents."""
        r = window_timing_reward(_base_ctx())
        assert r[0] == 0.0
        assert r[2] == 0.0


# ---------------------------------------------------------------------------
# assist_reward
# ---------------------------------------------------------------------------

class TestAssistReward:

    def test_assist_event_gives_reward_to_shield_agent(self):
        """window_assist[1]=True → blue0 (agent 2) gets R_ASSIST."""
        ctx = _base_ctx(window_assist=np.array([False, True]))
        r = assist_reward(ctx)
        assert abs(r[2] - R_ASSIST) < 1e-9, (
            f"blue0 expected {R_ASSIST}, got {r[2]}"
        )
        assert r[0] == 0.0, f"red0 expected 0, got {r[0]}"

    def test_no_assist_gives_zero(self):
        r = assist_reward(_base_ctx())
        assert r[0] == 0.0
        assert r[2] == 0.0


# ---------------------------------------------------------------------------
# pellet_hit_reward
# ---------------------------------------------------------------------------

class TestPelletHitReward:

    def test_three_hits_gives_correct_reward(self):
        """red1 (index 1) landed 3 pellets on opponent → 3 * R_PELLET_HIT."""
        hits = np.zeros(4, dtype=int)
        hits[1] = 3
        r = pellet_hit_reward(_base_ctx(pellet_hits_opp=hits))
        expected = 3 * R_PELLET_HIT
        assert abs(r[1] - expected) < 1e-9, f"red1 expected {expected}, got {r[1]}"

    def test_no_hits_gives_zero(self):
        r = pellet_hit_reward(_base_ctx())
        np.testing.assert_allclose(r, np.zeros(4))


# ---------------------------------------------------------------------------
# kill_reward
# ---------------------------------------------------------------------------

class TestKillReward:

    def test_kill_gives_reward_to_shotgun_agent(self):
        """red1 (index 1) scored 1 kill → R_KILL = +2.0."""
        kills = np.zeros(4, dtype=int)
        kills[1] = 1
        r = kill_reward(_base_ctx(kills=kills))
        assert abs(r[1] - R_KILL) < 1e-9, f"red1 expected {R_KILL}, got {r[1]}"

    def test_no_kill_gives_zero(self):
        r = kill_reward(_base_ctx())
        np.testing.assert_allclose(r, np.zeros(4))


# ---------------------------------------------------------------------------
# close_range_reward
# ---------------------------------------------------------------------------

class TestCloseRangeReward:

    def test_within_3m_gives_per_second_reward(self):
        """red1 (index 1) within 3m → R_CLOSE_RANGE * dt = 0.3 * 0.05 = 0.015."""
        within = np.zeros(4, dtype=bool)
        within[1] = True
        r = close_range_reward(_base_ctx(within_3m=within, dt=0.05))
        expected = R_CLOSE_RANGE * 0.05
        assert abs(r[1] - expected) < 1e-9, (
            f"red1 expected {expected} (R_CLOSE_RANGE * dt), got {r[1]}"
        )

    def test_outside_3m_gives_zero(self):
        r = close_range_reward(_base_ctx())
        np.testing.assert_allclose(r, np.zeros(4))


# ---------------------------------------------------------------------------
# shot_through_window_reward
# ---------------------------------------------------------------------------

class TestShotThroughWindowReward:

    def test_window_shot_gives_reward_to_shotgun(self):
        """shot_through_window_4plus[0]=True → red1 (agent 1) gets R_WINDOW_4PLUS."""
        ctx = _base_ctx(shot_through_window_4plus=np.array([True, False]))
        r = shot_through_window_reward(ctx)
        assert abs(r[1] - R_WINDOW_4PLUS) < 1e-9, (
            f"red1 expected {R_WINDOW_4PLUS}, got {r[1]}"
        )
        assert r[3] == 0.0, f"blue1 expected 0, got {r[3]}"

    def test_no_window_shot_gives_zero(self):
        r = shot_through_window_reward(_base_ctx())
        np.testing.assert_allclose(r, np.zeros(4))


# ---------------------------------------------------------------------------
# friendly_fire_penalty
# ---------------------------------------------------------------------------

class TestFriendlyFirePenalty:

    def test_two_friendly_pellets_gives_correct_penalty(self):
        """red1 (index 1) hit 2 friendly pellets → 2 * R_FRIENDLY_FIRE = -0.60."""
        fp = np.zeros(4, dtype=int)
        fp[1] = 2
        r = friendly_fire_penalty(_base_ctx(friendly_pellets=fp))
        expected = 2 * R_FRIENDLY_FIRE
        assert abs(r[1] - expected) < 1e-9, f"red1 expected {expected}, got {r[1]}"

    def test_no_friendly_fire_gives_zero(self):
        r = friendly_fire_penalty(_base_ctx())
        np.testing.assert_allclose(r, np.zeros(4))


# ---------------------------------------------------------------------------
# die_first_penalty
# ---------------------------------------------------------------------------

class TestDieFirstPenalty:

    def test_dying_before_teammate_gives_penalty(self):
        """red0 (agent 0) died while teammate alive → R_DIE_FIRST = -1.0."""
        died = np.zeros(4, dtype=bool)
        died[0] = True
        r = die_first_penalty(_base_ctx(died_before_teammate=died))
        assert abs(r[0] - R_DIE_FIRST) < 1e-9, (
            f"red0 expected {R_DIE_FIRST}, got {r[0]}"
        )
        assert r[1] == 0.0, f"red1 expected 0, got {r[1]}"

    def test_not_dying_first_gives_zero(self):
        r = die_first_penalty(_base_ctx())
        np.testing.assert_allclose(r, np.zeros(4))
