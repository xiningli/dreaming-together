"""Scripted coordinator — protocol seeding through the REAL z_g channel.

Design §6 G5 amendment: all Stage-1/2 cues reach listeners through the
same pathway the learned coordinator will use, so the listener interface
learned under scripting is exactly the interface the learned coordinator
inherits. Condition C: real vocabulary tokens through the frozen token
table. Conditions A/B: the analogous hand-built 5-d bottleneck through the
frozen decoder. Both carry the same information (equivalent seeding is an
integrity constraint).

Message content (the communication the task causally needs — the shield
cannot see its shotgun's cooldown):
  pos 0: OPEN_WINDOW / CLOSE_WINDOW  (shotgun ready + in range)
  pos 1: READY / RELOADING           (shotgun cooldown state)
  pos 2: LEFT / FRONT / RIGHT        (nearest enemy bearing from shield)
  pos 3: NEAR / FAR                  (enemy inside shotgun range?)
  pos 4: EXPOSED / SAFE              (own C_r below/above 0.5)
  pos 5-7: PAD
"""
from __future__ import annotations

import numpy as np
import torch

from dreaming_together.coordinators.vocab import TOK, MSG_LEN
from dreaming_together.envs.combat_env import (
    CombatEnv, SHOTGUN_OF_TEAM, SHIELD_OF_TEAM, SHOTGUN_RANGE, TEAM_OF,
    PREFIXES,
)
from dreaming_together.envs.tank import hull_pos, hull_yaw


def _team_state(env: CombatEnv, team: int) -> dict:
    sg, sh = SHOTGUN_OF_TEAM[team], SHIELD_OF_TEAM[team]
    sg_pos = hull_pos(env.model, env.data, sg)[:2]
    sh_pos = hull_pos(env.model, env.data, sh)[:2]
    opps = [o for o in PREFIXES if TEAM_OF[o] != team and env.hp[o] > 0] or \
           [o for o in PREFIXES if TEAM_OF[o] != team]
    enemy = min(opps, key=lambda o: np.linalg.norm(
        hull_pos(env.model, env.data, o)[:2] - sg_pos))
    e_pos = hull_pos(env.model, env.data, enemy)[:2]
    v = e_pos - sh_pos
    bearing = np.arctan2(v[1], v[0]) - hull_yaw(env.model, env.data, sh)
    bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
    return {
        "ready": env.cooldown[sg] <= 0.35,
        "reloading": env.cooldown[sg] > 0.15,
        "in_range": np.linalg.norm(e_pos - sg_pos) < SHOTGUN_RANGE,
        "bearing": bearing,
        "exposed": env._team_cr(team) < 0.5,
    }


def scripted_tokens(env: CombatEnv, team: int) -> torch.Tensor:
    st = _team_state(env, team)
    toks = [TOK["PAD"]] * MSG_LEN
    toks[0] = TOK["OPEN_WINDOW"] if (st["ready"] and st["in_range"]) \
        else TOK["CLOSE_WINDOW"]
    toks[1] = TOK["RELOADING"] if st["reloading"] else TOK["READY"]
    if st["bearing"] > 0.35:
        toks[2] = TOK["LEFT"]
    elif st["bearing"] < -0.35:
        toks[2] = TOK["RIGHT"]
    else:
        toks[2] = TOK["FRONT"]
    toks[3] = TOK["NEAR"] if st["in_range"] else TOK["FAR"]
    toks[4] = TOK["EXPOSED"] if st["exposed"] else TOK["SAFE"]
    return torch.tensor([toks], dtype=torch.long)


def scripted_bottleneck(env: CombatEnv, team: int) -> torch.Tensor:
    """A/B seeding: the same five facts as a hand-built 5-d vector."""
    st = _team_state(env, team)
    return torch.tensor([[
        1.0 if (st["ready"] and st["in_range"]) else -1.0,
        1.0 if st["reloading"] else -1.0,
        float(np.clip(st["bearing"] / np.pi, -1, 1)),
        1.0 if st["in_range"] else -1.0,
        1.0 if st["exposed"] else -1.0,
    ]], dtype=torch.float32)
