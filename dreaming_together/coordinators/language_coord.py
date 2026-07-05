"""Language coordinator (condition C).

s_τ → MLP_in → memory; 4-layer transformer decoder (d=128, 4 heads) emits
MSG_LEN=8 tokens autoregressively from the 32-token vocabulary. Tokens map
to z_g through a FROZEN 32×32 embedding table: 8 positions × 32 dims
concatenated = z_g ∈ R^256. Freezing the table fixes channel semantics so
listeners seeded on the scripted protocol keep a stable interface when
the learned coordinator takes over (same principle as the embedding
coordinator's frozen decoder).

Bandwidth: 8 tokens × log2(32) = 40 bits/frame.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dreaming_together.coordinators.vocab import (
    VOCAB_SIZE, MSG_LEN, BITS_PER_TOKEN,
)

Z_DIM = 256
TOK_EMBED = Z_DIM // MSG_LEN      # 32


class LanguageCoordinator(nn.Module):
    def __init__(self, state_dim: int, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.mlp_in = nn.Sequential(
            nn.Linear(state_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.tok_in = nn.Embedding(VOCAB_SIZE + 1, d_model)   # +1 = BOS
        self.pos = nn.Parameter(torch.randn(MSG_LEN, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=256, batch_first=True,
            dropout=0.0)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.head = nn.Linear(d_model, VOCAB_SIZE)

        # frozen channel semantics: token id → 32-d chunk of z_g
        self.z_table = nn.Embedding(VOCAB_SIZE, TOK_EMBED)
        for p in self.z_table.parameters():
            p.requires_grad_(False)

    def bits(self) -> int:
        return MSG_LEN * BITS_PER_TOKEN        # 40

    def _step_logits(self, s: torch.Tensor,
                     prev: torch.Tensor) -> torch.Tensor:
        """Logits for the next position given tokens so far.
        prev: (B, t) token ids (t ≥ 0); s: (B, state_dim)."""
        B, t = prev.shape[0], prev.shape[1]
        bos = torch.full((B, 1), VOCAB_SIZE, dtype=torch.long,
                         device=s.device)
        seq = torch.cat([bos, prev], dim=1)
        x = self.tok_in(seq) + self.pos[:t + 1]
        mask = nn.Transformer.generate_square_subsequent_mask(
            t + 1, device=s.device)
        mem = self.mlp_in(s).unsqueeze(1)
        h = self.decoder(x, mem, tgt_mask=mask)
        return self.head(h[:, -1])

    def forward(self, s: torch.Tensor, sample: bool = True):
        """Autoregressive emission. Returns
        (z_g, tokens (B, MSG_LEN), log_prob (B,), entropy (B,))."""
        B = s.shape[0]
        toks = torch.zeros(B, 0, dtype=torch.long, device=s.device)
        logp = torch.zeros(B, device=s.device)
        ent = torch.zeros(B, device=s.device)
        for _ in range(MSG_LEN):
            logits = self._step_logits(s, toks)
            dist = torch.distributions.Categorical(logits=logits)
            tok = dist.sample() if sample else logits.argmax(-1)
            logp = logp + dist.log_prob(tok)
            ent = ent + dist.entropy()
            toks = torch.cat([toks, tok.unsqueeze(1)], dim=1)
        return self.z_from_tokens(toks), toks, logp, ent

    def log_prob(self, s: torch.Tensor, toks: torch.Tensor):
        """Log-prob + entropy of a given message under the current policy
        (PPO update path)."""
        logp = torch.zeros(len(s), device=s.device)
        ent = torch.zeros(len(s), device=s.device)
        for t in range(MSG_LEN):
            logits = self._step_logits(s, toks[:, :t])
            dist = torch.distributions.Categorical(logits=logits)
            logp = logp + dist.log_prob(toks[:, t])
            ent = ent + dist.entropy()
        return logp, ent

    def z_from_tokens(self, toks: torch.Tensor) -> torch.Tensor:
        """Fixed channel semantics: (B, MSG_LEN) ids → (B, 256)."""
        return self.z_table(toks).flatten(1)
