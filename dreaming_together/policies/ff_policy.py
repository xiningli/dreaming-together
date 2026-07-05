"""Feedforward policy (condition A building block) and Gaussian head for RL.

Small on purpose: the T0/T1 ladder rungs prove the training loop, not
capacity. The condition-A policy for the full experiment (MLP over the
fused conditioning vector, 8×5 action horizon) will extend this module.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """tanh-mean Gaussian policy with state-independent learnable log-std."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: tuple[int, ...] = (64, 64),
                 init_log_std: float = -0.5):
        super().__init__()
        self.mean_net = MLP(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = torch.tanh(self.mean_net(obs))
        return torch.distributions.Normal(mean, self.log_std.exp())

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1)
