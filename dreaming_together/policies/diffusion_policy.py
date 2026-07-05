"""Diffusion policy (conditions B and C) — DDPM training, DDIM sampling.

Faithful to design §3.2: action horizon a⁰ ∈ R^{H×A} in [-1,1], cosine
noise schedule with K=100, ε-prediction loss, 8-step DDIM at deployment,
EMA weights for evaluation.

Denoiser deviation, documented: the design specifies a 1-D temporal U-Net
over the horizon axis. At H=8 a 4-level U-Net collapses (8→4→2→1), so the
denoiser here is a FiLM-conditioned MLP over the flattened horizon —
equivalent capacity at this horizon length. Revisit if H grows.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn


def cosine_alpha_bar(K: int, s: float = 0.008) -> torch.Tensor:
    """ᾱ_k for k = 0..K (Nichol & Dhariwal cosine schedule)."""
    k = torch.arange(K + 1, dtype=torch.float64)
    f = torch.cos(((k / K) + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-5, 1.0)
    return ab.float()


class _FiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return torch.relu(self.lin(x) * (1 + scale) + shift)


class DiffusionPolicy(nn.Module):
    """ε-prediction denoiser over a flattened (H × A) action horizon."""

    def __init__(self, cond_dim: int, act_dim: int = 5, horizon: int = 8,
                 hidden: int = 256, K: int = 100, k_embed: int = 64):
        super().__init__()
        self.cond_dim, self.act_dim, self.horizon = cond_dim, act_dim, horizon
        self.K = K
        self.x_dim = act_dim * horizon
        self.register_buffer("alpha_bar", cosine_alpha_bar(K))

        half = k_embed // 2
        freqs = torch.exp(torch.linspace(0, math.log(1000), half))
        self.register_buffer("freqs", freqs)
        film_dim = cond_dim + k_embed

        self.inp = nn.Linear(self.x_dim, hidden)
        self.blocks = nn.ModuleList(
            [_FiLMBlock(hidden, film_dim) for _ in range(3)])
        self.out = nn.Linear(hidden, self.x_dim)

    # -- conditioning ------------------------------------------------------
    def _k_embed(self, k: torch.Tensor) -> torch.Tensor:
        ang = k.float().unsqueeze(-1) / self.K * self.freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def eps(self, x_k: torch.Tensor, k: torch.Tensor,
            c: torch.Tensor) -> torch.Tensor:
        """Predict ε from noised horizon x_k at step k, condition c."""
        cond = torch.cat([c, self._k_embed(k)], dim=-1)
        h = torch.relu(self.inp(x_k))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(h)

    # -- training ------------------------------------------------------------
    def loss(self, a0: torch.Tensor, c: torch.Tensor,
             weights: torch.Tensor | None = None) -> torch.Tensor:
        """ε-prediction MSE. a0: (B, H*A) in [-1,1]; c: (B, cond_dim).
        weights: optional (B,) per-sample weights (AWR fine-tuning)."""
        B = len(a0)
        k = torch.randint(1, self.K + 1, (B,), device=a0.device)
        ab = self.alpha_bar[k].unsqueeze(-1)
        noise = torch.randn_like(a0)
        x_k = ab.sqrt() * a0 + (1 - ab).sqrt() * noise
        err = (self.eps(x_k, k, c) - noise) ** 2
        per = err.mean(dim=-1)
        if weights is not None:
            per = per * weights
        return per.mean()

    # -- sampling ------------------------------------------------------------
    @torch.no_grad()
    def ddim_sample(self, c: torch.Tensor, n_steps: int = 8,
                    noise_scale: float = 1.0,
                    generator: torch.Generator | None = None) -> torch.Tensor:
        """Deterministic DDIM (η=0) from n_steps evenly spaced k's.
        Returns (B, H*A) clipped to [-1,1]. noise_scale scales the initial
        latent (0 → mode-seeking, 1 → full diversity)."""
        B = len(c)
        ks = torch.linspace(self.K, 0, n_steps + 1).round().long()
        x = torch.randn(B, self.x_dim, generator=generator,
                        device=c.device) * noise_scale
        for i in range(n_steps):
            k, k_next = ks[i], ks[i + 1]
            ab = self.alpha_bar[k]
            e = self.eps(x, torch.full((B,), k, device=c.device), c)
            x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)
            ab_n = self.alpha_bar[k_next]
            x = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * e
        return x.clamp(-1, 1)

    @torch.no_grad()
    def ddpm_sample(self, c: torch.Tensor,
                    generator: torch.Generator | None = None) -> torch.Tensor:
        """Full K-step ancestral DDPM sampling (reference for the G4
        DDIM-fidelity check)."""
        B = len(c)
        x = torch.randn(B, self.x_dim, generator=generator, device=c.device)
        for k in range(self.K, 0, -1):
            ab, ab_p = self.alpha_bar[k], self.alpha_bar[k - 1]
            alpha = ab / ab_p
            e = self.eps(x, torch.full((B,), k, device=c.device), c)
            x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)
            mean = (alpha.sqrt() * (1 - ab_p) * x
                    + ab_p.sqrt() * (1 - alpha) * x0) / (1 - ab)
            if k > 1:
                sigma = ((1 - ab_p) / (1 - ab) * (1 - alpha)).sqrt()
                x = mean + sigma * torch.randn(B, self.x_dim,
                                               generator=generator,
                                               device=c.device)
            else:
                x = mean
        return x.clamp(-1, 1)

    def horizon_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, self.horizon, self.act_dim)


class EMA:
    """Exponential moving average of model weights (design: 0.9999)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p, 1.0 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)
