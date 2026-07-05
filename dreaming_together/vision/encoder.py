"""Segmentation encoder — the shared visual front-end (design §3.3, §3.7).

Input:  (B, 64, 64) int8 class ids, 6 classes (envs/cameras.py).
Encode: one-hot → 4 strided conv blocks → 256-d embedding.
Decode: mirror deconvs → (B, 6, 64, 64) class logits, used only for the
Stage 0 reconstruction pretraining; policies consume the embedding.

Smaller than the design's ~5M ResNet-10 (≈1.6M params) — the input is
already a clean 6-class map, not RGB; capacity goes further here. Revisit
if Stage-1-on-vision parity fails.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dreaming_together.envs.cameras import N_SEG_CLASSES

EMBED_DIM = 256


class SegEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        c = N_SEG_CLASSES
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1), nn.ReLU(),    # 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),  # 8
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(), # 4
        )
        self.fc = nn.Linear(128 * 4 * 4, EMBED_DIM)

    def forward(self, seg: torch.Tensor) -> torch.Tensor:
        """seg: (B, 64, 64) int64/int8 class ids → (B, 256) embedding."""
        x = torch.nn.functional.one_hot(
            seg.long(), N_SEG_CLASSES).permute(0, 3, 1, 2).float()
        return self.fc(self.conv(x).flatten(1))


class SegDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(EMBED_DIM, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, N_SEG_CLASSES, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconv(self.fc(z).view(-1, 128, 4, 4))
