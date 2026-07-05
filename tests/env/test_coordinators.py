"""Coordinator tests: bandwidth parity, shapes, channel semantics frozen,
quantization straight-through gradients, scripted seeding produces valid
messages through the real channel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.coordinators.bandwidth import assert_bandwidth_parity
from dreaming_together.coordinators.embedding_coord import (
    EmbeddingCoordinator, quantize,
)
from dreaming_together.coordinators.language_coord import LanguageCoordinator
from dreaming_together.coordinators.vocab import MSG_LEN, VOCAB_SIZE

STATE_DIM = 59


class TestBandwidth:

    def test_parity_is_40_bits(self):
        assert assert_bandwidth_parity() == 40


class TestEmbeddingCoordinator:

    def test_shapes_and_quantization(self):
        c = EmbeddingCoordinator(STATE_DIM)
        s = torch.randn(6, STATE_DIM)
        z, b, logp, ent = c(s)
        assert z.shape == (6, 256) and b.shape == (6, 5)
        assert logp.shape == (6,) and ent.shape == (6,)
        # quantization: 256 levels, straight-through grad
        x = torch.linspace(-1, 1, 11, requires_grad=True)
        q = quantize(x)
        assert len(torch.unique(q)) <= 11
        q.sum().backward()
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_decoder_frozen(self):
        c = EmbeddingCoordinator(STATE_DIM)
        assert all(not p.requires_grad for p in c.dec.parameters())


class TestLanguageCoordinator:

    def test_emission_shapes_and_determinism(self):
        c = LanguageCoordinator(STATE_DIM)
        s = torch.randn(4, STATE_DIM)
        z, toks, logp, ent = c(s, sample=False)
        assert z.shape == (4, 256)
        assert toks.shape == (4, MSG_LEN)
        assert toks.max() < VOCAB_SIZE
        z2, toks2, _, _ = c(s, sample=False)
        assert torch.equal(toks, toks2), "greedy emission must be deterministic"

    def test_log_prob_matches_sampling(self):
        torch.manual_seed(0)
        c = LanguageCoordinator(STATE_DIM)
        s = torch.randn(3, STATE_DIM)
        _, toks, logp_s, _ = c(s, sample=True)
        logp_r, ent = c.log_prob(s, toks)
        assert torch.allclose(logp_s, logp_r, atol=1e-4)

    def test_z_table_frozen(self):
        c = LanguageCoordinator(STATE_DIM)
        assert not c.z_table.weight.requires_grad


class TestScriptedSeeding:

    def test_scripted_messages_through_real_channel(self):
        from dreaming_together.envs.combat_env import CombatEnv
        from dreaming_together.oracle.scripted_coordinator import (
            scripted_tokens, scripted_bottleneck,
        )
        env = CombatEnv(seed=0, privileged_obs=True)
        env.reset(seed=3)
        toks = scripted_tokens(env, 0)
        assert toks.shape == (1, MSG_LEN) and toks.max() < VOCAB_SIZE
        lang = LanguageCoordinator(STATE_DIM)
        z = lang.z_from_tokens(toks)
        assert z.shape == (1, 256) and torch.isfinite(z).all()

        b = scripted_bottleneck(env, 0)
        emb = EmbeddingCoordinator(STATE_DIM)
        z2 = emb.z_from_bottleneck(b)
        assert z2.shape == (1, 256) and torch.isfinite(z2).all()
