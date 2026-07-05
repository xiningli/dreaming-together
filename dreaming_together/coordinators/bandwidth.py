"""Bandwidth parity assertion (integrity constraint, design §8).

Called at the top of every training/eval entry point that involves
coordinators. Conditions must never differ in bits/frame.
"""
from __future__ import annotations

from dreaming_together.coordinators.vocab import bits_per_message
from dreaming_together.coordinators.embedding_coord import (
    D_BOTTLENECK, QUANT_BITS,
)

BITS = 40


def assert_bandwidth_parity() -> int:
    bits_lang = bits_per_message()
    bits_embed = D_BOTTLENECK * QUANT_BITS
    assert bits_lang == bits_embed == BITS, (
        f"bandwidth mismatch: lang={bits_lang} embed={bits_embed} "
        f"expected={BITS}")
    return BITS
