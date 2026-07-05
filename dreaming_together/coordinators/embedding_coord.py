"""Embedding coordinator (conditions A and B).

s_τ → MLP encoder → 5-d tanh bottleneck → 8-bit/dim quantization
(straight-through) → FROZEN decoder → z_g ∈ R^256.

The bottleneck→z_g decoder is frozen at init (matching the language
coordinator's frozen token-embedding table): the coordinator learns WHAT
to send, never what the channel symbols mean — so listeners seeded on the
scripted protocol keep a stable interface when the learned coordinator
takes over (protocol-seeding amendment, design §6 G5).

Bandwidth: 5 dims × 8 bits = 40 bits/frame.
"""
from __future__ import annotations

import torch
import torch.nn as nn

D_BOTTLENECK = 5
QUANT_BITS = 8
Z_DIM = 256


def quantize(x: torch.Tensor) -> torch.Tensor:
    """8-bit/dim quantization on [-1,1] with straight-through gradients."""
    levels = 2 ** QUANT_BITS - 1
    q = torch.round((x + 1) / 2 * levels) / levels * 2 - 1
    return x + (q - x).detach()


class EmbeddingCoordinator(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, D_BOTTLENECK), nn.Tanh(),
        )
        # exploration noise (PPO acts on the pre-quantization bottleneck)
        self.log_std = nn.Parameter(torch.full((D_BOTTLENECK,), -1.0))
        self.dec = nn.Sequential(
            nn.Linear(D_BOTTLENECK, Z_DIM), nn.Tanh())
        for p in self.dec.parameters():        # frozen channel decoder
            p.requires_grad_(False)

    def bits(self) -> int:
        return D_BOTTLENECK * QUANT_BITS       # 40

    def forward(self, s: torch.Tensor, sample: bool = True):
        """Returns (z_g, bottleneck_action, log_prob, entropy)."""
        mu = self.enc(s)
        dist = torch.distributions.Normal(mu, self.log_std.exp())
        b = dist.sample() if sample else mu
        b = b.clamp(-1, 1)
        z = self.dec(quantize(b))
        return z, b, dist.log_prob(b).sum(-1), dist.entropy().sum(-1)

    def z_from_bottleneck(self, b: torch.Tensor) -> torch.Tensor:
        """Scripted/seeding path: a hand-built 5-d vector → z_g."""
        return self.dec(quantize(b.clamp(-1, 1)))
