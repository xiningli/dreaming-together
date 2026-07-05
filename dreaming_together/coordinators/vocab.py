"""Coordinator vocabulary and message format (condition C).

Bandwidth-exact deviation from the prose design, documented: the design
says "40 tokens, 40 bits/frame", but log2(40) is irrational and the
integrity constraint (bits_lang == bits_embed == 40, asserted at startup)
outranks vocab-size fidelity. Here: 32 tokens × 8 positions = 5 bits × 8
= exactly 40 bits/frame, matching the embedding coordinator's 5 dims ×
8-bit quantization exactly.
"""
from __future__ import annotations

TOKENS = [
    "PAD",
    # window / fire coordination
    "OPEN_WINDOW", "CLOSE_WINDOW", "FIRE", "CEASE", "READY", "RELOADING",
    "HARD", "SOFT",
    # maneuver
    "ADVANCE", "HOLD", "RETREAT", "FLANK_LEFT", "FLANK_RIGHT", "COVER_ME",
    # referents
    "ENEMY", "ALLY", "SHIELD", "SHOTGUN",
    # spatial
    "NEAR", "FAR", "LEFT", "RIGHT", "FRONT", "BEHIND", "WALL", "CORNER",
    # status
    "EXPOSED", "SAFE", "BLOCKED", "WINDOW", "HURT",
]
assert len(TOKENS) == 32

VOCAB_SIZE = 32
MSG_LEN = 8
BITS_PER_TOKEN = 5          # log2(32)
TOK = {name: i for i, name in enumerate(TOKENS)}


def bits_per_message() -> int:
    return MSG_LEN * BITS_PER_TOKEN     # 40
