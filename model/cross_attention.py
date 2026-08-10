"""
cross_attention.py
-------------------
Bahdanau-style additive attention for the genome error-correction Seq2Seq decoder.

At each decoding step t, given the decoder's previous hidden state s_{t-1} and the
fused encoder hidden states h_i (i = 1..L), this module computes:

    e_{t,i}     = v^T tanh(W_enc @ h_i + W_dec @ s_{t-1})
    alpha_{t,i} = softmax_i(e_{t,i})            <-- masked so <PAD> positions get -inf
    c_t         = sum_i alpha_{t,i} * h_i

`alpha` (shape (B, L)) is returned alongside the context vector on every single
call. The decoder is responsible for stacking `alpha` across all T decode steps
into the full (B, T, L) alignment matrix that gets serialized by the FastAPI
backend for the frontend XAI heatmap -- this module just produces one row of
that matrix per call.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    # encoder_dim MUST match HybridEncoder's cfg.fused_dim (the last dim of
    # encoder_outputs). decoder_dim MUST match the decoder LSTMCell's hidden size.
    encoder_dim: int = 512
    decoder_dim: int = 512
    attn_dim: int = 256


class BahdanauAttention(nn.Module):
    """
    Additive attention with the encoder-side projection factored out into
    `project_encoder_outputs()`.

    Why factor it out: W_enc @ h_i does not depend on the decoder's timestep,
    only on the (fixed, already-computed) encoder_outputs. A naive implementation
    that recomputes W_enc @ h_i inside forward() on every one of the T decode
    steps redoes the same (B, L, attn_dim) matmul T times for no reason. For
    long prokaryotic reads chunked at, say, 1024 tokens, that's 1024 wasted
    projections per chunk. Call `project_encoder_outputs()` once per batch,
    before the decode loop starts, and pass the cached result into forward()
    at every step.
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Linear(cfg.encoder_dim, cfg.attn_dim, bias=False)
        self.W_dec = nn.Linear(cfg.decoder_dim, cfg.attn_dim, bias=False)
        self.v = nn.Linear(cfg.attn_dim, 1, bias=False)

    def project_encoder_outputs(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        encoder_outputs: (B, L, encoder_dim) -> (B, L, attn_dim)

        Call ONCE per batch, outside the autoregressive decode loop.
        """
        return self.W_enc(encoder_outputs)

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # (B, decoder_dim)      -- s_{t-1}
        encoder_outputs: torch.Tensor,  # (B, L, encoder_dim)   -- h_i, used as attention VALUES
        encoder_proj: torch.Tensor,     # (B, L, attn_dim)      -- precomputed W_enc @ h_i
        mask: torch.Tensor,             # (B, L) bool, True = real (non-pad) token
    ):
        """
        Returns:
            context : (B, encoder_dim)  -- c_t, the weighted sum of encoder_outputs
            alpha   : (B, L)            -- the attention distribution for this single step
        """
        if encoder_proj.shape[:2] != encoder_outputs.shape[:2]:
            raise ValueError(
                f"encoder_proj batch/length {tuple(encoder_proj.shape[:2])} does not match "
                f"encoder_outputs {tuple(encoder_outputs.shape[:2])}. Did you forget to call "
                f"project_encoder_outputs(encoder_outputs) for this exact batch?"
            )

        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)  # (B, 1, attn_dim)
        energy = torch.tanh(encoder_proj + dec_proj)        # (B, L, attn_dim), broadcast over L
        scores = self.v(energy).squeeze(-1)                  # (B, L)

        # Additive -inf mask: padding positions must NEVER receive attention mass,
        # otherwise the decoder can learn to "attend" to noise that carries no
        # biological signal, and the XAI heatmap will show bogus focus regions
        # past the true end of a read.
        scores = scores.masked_fill(~mask, float("-inf"))

        alpha = F.softmax(scores, dim=-1)  # (B, L)

        # Edge case: if a row of `mask` is all-False (a zero-length sequence
        # slipped through), every score is -inf and softmax produces NaN, not
        # zero. That would silently poison the context vector and downstream
        # loss. Guard against it explicitly rather than let it propagate.
        alpha = torch.nan_to_num(alpha, nan=0.0)

        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, encoder_dim)

        return context, alpha
