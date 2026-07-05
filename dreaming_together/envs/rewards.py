"""Per-step reward computation for the 2v2 tank combat environment.

Agent indexing (fixed by convention)
-------------------------------------
  0  red0   — shield
  1  red1   — shotgun
  2  blue0  — shield
  3  blue1  — shotgun

Team indexing
  0  red  (agents 0, 1)
  1  blue (agents 2, 3)

Shield rewards apply to agents at indices [0, 2].
Shotgun rewards apply to agents at indices [1, 3].

Reward constants
----------------
All constants are documented with their units.  Functions raise
NotImplementedError until implemented.  Tests call the individual functions
directly so each reward component can be validated in isolation.

Advance reward
--------------
R_ADVANCE units: reward per metre of distance closed per policy step.
Formula: R_ADVANCE * max(0, dist_prev - dist_now)

Close-range reward (shotgun only)
----------------------------------
R_CLOSE_RANGE units: reward per second while within 3 m.
Per step: R_CLOSE_RANGE * ctx.dt
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

R_TIME_ALIVE     =  0.005   # per step, all living agents
R_DAMAGE_TAKEN   = -0.02    # per HP of damage taken
R_ADVANCE        =  0.002   # per metre closed this step
R_WIN            = +10.0    # terminal: winning team
R_LOSS           = -10.0    # terminal: losing team
R_DRAW           =  -5.0    # terminal: both teams

# Shield-specific
R_SHIELD_BLOCKS  =  0.02    # per step while shield blocks opponent line-of-fire
R_WINDOW_GOOD    =  0.05    # per step: window open AND opponent in reload
R_WINDOW_BAD     = -0.05    # per step: window open AND opponent ready to fire
R_ASSIST         =  1.5     # per assist event (teammate scored through open window)
R_DIE_FIRST      = -1.0     # dying while teammate still alive (shield and shotgun)

# Shotgun-specific
R_PELLET_HIT     =  0.5     # per pellet that hits an opponent body
R_KILL           =  2.0     # per kill
R_CLOSE_RANGE    =  0.3     # per second while within 3 m of nearest opponent
R_WINDOW_4PLUS   =  0.5     # per shot through open window landing >= 4 pellets
R_FRIENDLY_FIRE  = -0.3     # per pellet that hits a teammate body

LAM_AR           = 1e-3     # action-rate penalty coefficient


# ---------------------------------------------------------------------------
# Episode state context
# ---------------------------------------------------------------------------

@dataclass
class RewardContext:
    """All per-step episode state variables needed to compute rewards.

    Shapes: N=4 (all agents), T=2 (red=0, blue=1 teams).
    """
    dt: float                            # policy step duration (seconds)

    # --- Common ---
    alive: np.ndarray                    # (N,) bool
    damage_taken: np.ndarray             # (N,) float, HP lost this step (>= 0)
    killed_this_step: np.ndarray         # (N,) bool, True if died exactly this step
    dist_to_nearest_opp: np.ndarray      # (N,) float, metres
    prev_dist_to_nearest_opp: np.ndarray # (N,) float, metres, previous step
    action: np.ndarray                   # (N, 5) current normalised action
    prev_action: np.ndarray              # (N, 5) previous normalised action
    episode_done: bool                   # True on terminal step
    team_result: np.ndarray              # (T,) int: +1 win, -1 loss, 0 draw. meaningful only when episode_done

    # --- Shield-specific (per team) ---
    shield_blocks_line: np.ndarray       # (T,) bool: shield is blocking opponent line-of-fire
    window_open: np.ndarray              # (T,) bool: W_q > threshold this step
    opp_in_reload: np.ndarray            # (T,) bool: nearest opponent is in reload cooldown
    window_assist: np.ndarray            # (T,) bool: teammate fired through <1.5 s old window AND pellets landed

    # --- Shotgun-specific (per agent) ---
    pellet_hits_opp: np.ndarray          # (N,) int, pellets hitting any opponent this step
    friendly_pellets: np.ndarray         # (N,) int, pellets hitting any teammate this step
    kills: np.ndarray                    # (N,) int, opponents killed by this agent this step
    within_3m: np.ndarray                # (N,) bool
    shot_through_window_4plus: np.ndarray  # (T,) bool: team's shotgun fired through window landing >= 4 pellets
    died_before_teammate: np.ndarray     # (N,) bool: this agent died while teammate still alive

    lam_ar: float = LAM_AR


# ---------------------------------------------------------------------------
# Reward component functions
# Each returns a (4,) float array of per-agent rewards for that component.
# ---------------------------------------------------------------------------

def time_alive_reward(ctx: RewardContext) -> np.ndarray:
    """R_TIME_ALIVE for each living agent per step."""
    return np.where(ctx.alive, R_TIME_ALIVE, 0.0).astype(float)


def damage_taken_penalty(ctx: RewardContext) -> np.ndarray:
    """R_DAMAGE_TAKEN * HP_lost for each agent."""
    return (R_DAMAGE_TAKEN * ctx.damage_taken).astype(float)


def advance_reward(ctx: RewardContext) -> np.ndarray:
    """R_ADVANCE * max(0, dist_prev - dist_now) per agent."""
    delta = ctx.prev_dist_to_nearest_opp - ctx.dist_to_nearest_opp
    return (R_ADVANCE * np.maximum(0.0, delta)).astype(float)


def action_rate_penalty(ctx: RewardContext) -> np.ndarray:
    """-lam_ar * ||a_t - a_{t-1}||^2 per agent."""
    diff = ctx.action - ctx.prev_action
    return (-ctx.lam_ar * np.sum(diff ** 2, axis=1)).astype(float)


def result_reward(ctx: RewardContext) -> np.ndarray:
    """R_WIN / R_LOSS / R_DRAW for each agent on a terminal step, else 0."""
    r = np.zeros(4)
    if not ctx.episode_done:
        return r
    # team_result[0]=red, team_result[1]=blue
    for agent_idx in range(4):
        team = 0 if agent_idx < 2 else 1
        result = ctx.team_result[team]
        if result == 1:
            r[agent_idx] = R_WIN
        elif result == -1:
            r[agent_idx] = R_LOSS
        else:
            r[agent_idx] = R_DRAW
    return r


def shield_blocking_reward(ctx: RewardContext) -> np.ndarray:
    """R_SHIELD_BLOCKS for shield agents (indices 0, 2) while blocking.

    shield_blocks_line[t] -> reward for agent t*2 (red0 if t=0, blue0 if t=1).
    """
    r = np.zeros(4)
    r[0] = R_SHIELD_BLOCKS if ctx.shield_blocks_line[0] else 0.0
    r[2] = R_SHIELD_BLOCKS if ctx.shield_blocks_line[1] else 0.0
    return r


def window_timing_reward(ctx: RewardContext) -> np.ndarray:
    """R_WINDOW_GOOD or R_WINDOW_BAD for shield agents based on opponent state.

    window_open[t] AND opp_in_reload[t] -> R_WINDOW_GOOD for agent t*2.
    window_open[t] AND NOT opp_in_reload[t] -> R_WINDOW_BAD for agent t*2.
    """
    r = np.zeros(4)
    for t, agent_idx in enumerate([0, 2]):
        if ctx.window_open[t]:
            r[agent_idx] = R_WINDOW_GOOD if ctx.opp_in_reload[t] else R_WINDOW_BAD
    return r


def assist_reward(ctx: RewardContext) -> np.ndarray:
    """R_ASSIST for shield agents when a window assist event fires.

    window_assist[t] -> R_ASSIST for agent t*2.
    """
    r = np.zeros(4)
    r[0] = R_ASSIST if ctx.window_assist[0] else 0.0
    r[2] = R_ASSIST if ctx.window_assist[1] else 0.0
    return r


def pellet_hit_reward(ctx: RewardContext) -> np.ndarray:
    """R_PELLET_HIT * pellets_on_opponent for shotgun agents (indices 1, 3)."""
    r = np.zeros(4)
    r[1] = R_PELLET_HIT * ctx.pellet_hits_opp[1]
    r[3] = R_PELLET_HIT * ctx.pellet_hits_opp[3]
    return r


def kill_reward(ctx: RewardContext) -> np.ndarray:
    """R_KILL * kills for shotgun agents (indices 1, 3)."""
    r = np.zeros(4)
    r[1] = R_KILL * ctx.kills[1]
    r[3] = R_KILL * ctx.kills[3]
    return r


def close_range_reward(ctx: RewardContext) -> np.ndarray:
    """R_CLOSE_RANGE * dt for shotgun agents (indices 1, 3) within 3 m."""
    r = np.zeros(4)
    r[1] = R_CLOSE_RANGE * ctx.dt if ctx.within_3m[1] else 0.0
    r[3] = R_CLOSE_RANGE * ctx.dt if ctx.within_3m[3] else 0.0
    return r


def shot_through_window_reward(ctx: RewardContext) -> np.ndarray:
    """R_WINDOW_4PLUS for shotgun agents when >= 4 pellets landed through open window.

    shot_through_window_4plus[t] -> R_WINDOW_4PLUS for agent t*2+1.
    """
    r = np.zeros(4)
    r[1] = R_WINDOW_4PLUS if ctx.shot_through_window_4plus[0] else 0.0
    r[3] = R_WINDOW_4PLUS if ctx.shot_through_window_4plus[1] else 0.0
    return r


def friendly_fire_penalty(ctx: RewardContext) -> np.ndarray:
    """R_FRIENDLY_FIRE * friendly_pellets for shotgun agents (indices 1, 3)."""
    r = np.zeros(4)
    r[1] = R_FRIENDLY_FIRE * ctx.friendly_pellets[1]
    r[3] = R_FRIENDLY_FIRE * ctx.friendly_pellets[3]
    return r


def die_first_penalty(ctx: RewardContext) -> np.ndarray:
    """R_DIE_FIRST for any agent that died this step while their teammate was alive."""
    return np.where(ctx.died_before_teammate, R_DIE_FIRST, 0.0).astype(float)


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

def compute_rewards(ctx: RewardContext) -> np.ndarray:
    """Compute and sum all per-agent reward components. Returns (4,) float array."""
    return (
        time_alive_reward(ctx)
        + damage_taken_penalty(ctx)
        + advance_reward(ctx)
        + action_rate_penalty(ctx)
        + result_reward(ctx)
        + shield_blocking_reward(ctx)
        + window_timing_reward(ctx)
        + assist_reward(ctx)
        + pellet_hit_reward(ctx)
        + kill_reward(ctx)
        + close_range_reward(ctx)
        + shot_through_window_reward(ctx)
        + friendly_fire_penalty(ctx)
        + die_first_penalty(ctx)
    )
