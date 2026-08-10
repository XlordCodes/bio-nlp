"""
rle_decoder.py
---------------
Auto-regressive LSTMCell decoder for the genome error-correction Seq2Seq model.

Two things this file is specifically responsible for, per the project's locked
design decisions:

1. RLE INTEGRATION
   Stride-1 hexamer tokens make every token inside a homopolymer run (e.g. the
   run of A's in ...GGAAAAAATCC...) look identical regardless of how long the
   run actually is -- the run length is only implicit in *how many* repeated
   tokens appear, not in any single token's identity. That's exactly the signal
   we need to reconstruct correctly, so it is fed to the decoder explicitly,
   out-of-band from the k-mer embeddings, as a per-source-position
   (base, run_length) auxiliary channel. See `RLEFeatureEmbedder` below.

   IMPORTANT UPSTREAM CONTRACT: this file does not compute run-length-encoding
   itself. It expects `rle_base_ids` and `rle_run_lengths`, each of shape
   (B, L) aligned 1:1 with `encoder_outputs` positions, to already be computed
   by data/tokenizer.py (or data/dataset.py) from the raw nucleotide string and
   passed in here. That module must produce, for source position i: the base
   at that position (A/C/G/T/N) and the length of the homopolymer run it
   belongs to.

2. ATTENTION MATRIX EXTRACTION FOR XAI
   `BahdanauAttention` returns one alignment row (B, L) per decode step. This
   file stacks all T of those rows into the full (B, T, L) alignment matrix
   and returns it directly from forward(), unmodified, so
   backend/inference_engine.py can serialize it as-is for the frontend
   AttentionHeatmap component. Nothing here truncates, averages, or
   downsamples it -- if you want a lower-resolution heatmap, that resizing
   belongs in the backend/frontend layer, not silently baked in here.

Zero-truncation note: this file is complete and runnable as-is (see the
`__main__` integration test at the bottom, which chains a real HybridEncoder
into this decoder end-to-end).
"""

import random
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cross_attention import AttentionConfig, BahdanauAttention
from config import (
    VOCAB_SIZE,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    RLE_BASE_TO_IDX,
    RLE_PAD_BASE_IDX,
    RLE_BASE_VOCAB_SIZE,
    RLE_DEFAULT_MAX_RUN_LENGTH,
)
# NOTE: RLE_BASE_TO_IDX / RLE_PAD_BASE_IDX / RLE_BASE_VOCAB_SIZE used to be
# defined locally in this file, and VOCAB_SIZE/PAD_IDX/SOS_IDX/EOS_IDX used
# to be imported from model.hybrid_encoder. Both now come from config.py
# directly -- the single source of truth per the project manifest -- so
# data/tokenizer.py can depend on config.py alone rather than reaching into
# model/, which would invert the manifest's intended dependency direction.


@dataclass
class RLEConfig:
    base_vocab_size: int = RLE_BASE_VOCAB_SIZE
    max_run_length: int = RLE_DEFAULT_MAX_RUN_LENGTH  # homopolymer runs longer than this are clamped, not dropped
    base_embed_dim: int = 16
    length_embed_dim: int = 16

    @property
    def output_dim(self) -> int:
        return self.base_embed_dim + self.length_embed_dim


class RLEFeatureEmbedder(nn.Module):
    """
    Embeds the (base, run_length) pair at every source position into a dense
    vector. Base identity and run length are embedded separately (each captures
    a different kind of signal -- "which base" vs "how long is this run") and
    concatenated, rather than combined into one joint embedding table, which
    would need base_vocab_size * max_run_length rows for no real benefit.
    """

    def __init__(self, cfg: RLEConfig):
        super().__init__()
        self.cfg = cfg
        self.base_embedding = nn.Embedding(
            cfg.base_vocab_size, cfg.base_embed_dim, padding_idx=RLE_PAD_BASE_IDX
        )
        # +1 because run lengths are clamped into the inclusive range [0, max_run_length]
        self.length_embedding = nn.Embedding(cfg.max_run_length + 1, cfg.length_embed_dim)

    def forward(self, base_ids: torch.Tensor, run_lengths: torch.Tensor) -> torch.Tensor:
        """
        base_ids    : (B, L) int64, values in [0, base_vocab_size)
        run_lengths : (B, L) int64, raw homopolymer run counts (>= 0)

        Returns: (B, L, cfg.output_dim)
        """
        clamped_lengths = torch.clamp(run_lengths, min=0, max=self.cfg.max_run_length)
        base_emb = self.base_embedding(base_ids)          # (B, L, base_embed_dim)
        length_emb = self.length_embedding(clamped_lengths)  # (B, L, length_embed_dim)
        return torch.cat([base_emb, length_emb], dim=-1)  # (B, L, output_dim)


@dataclass
class DecoderConfig:
    vocab_size: int = VOCAB_SIZE     # SAME k-mer vocab as the encoder (source and target share it)
    embed_dim: int = 256             # target token embedding dim
    hidden_dim: int = 512            # LSTMCell hidden size -- MUST match HybridEncoder cfg.fused_dim
    pad_idx: int = PAD_IDX
    sos_idx: int = SOS_IDX
    eos_idx: int = EOS_IDX
    dropout: float = 0.3
    attention: AttentionConfig = field(
        default_factory=lambda: AttentionConfig(encoder_dim=512, decoder_dim=512, attn_dim=256)
    )
    rle: RLEConfig = field(default_factory=RLEConfig)


class RLEDecoder(nn.Module):
    """
    Auto-regressive decoder. At each step t it consumes:
        - the embedding of the previous target token y_{t-1}
        - the attention context vector c_t (pooled over encoder_outputs)
        - the RLE context vector (pooled over the RLE embeddings using the
          SAME attention weights alpha, so whatever source position(s) the
          decoder is focusing on, it gets both the encoded sequence content
          AND the homopolymer run-length signal at that position)

    and produces a distribution over the vocabulary for y_t.

    forward() returns:
        logits            : (B, T, vocab_size)  -- raw scores, pass to CrossEntropyLoss
        predicted_tokens  : (B, T)               -- argmax token ids at each step
        attention_matrix  : (B, T, L)            -- full alignment matrix, alpha stacked over all T steps

    Loss masking contract: target_tokens is expected to be right-padded with
    `pad_idx` past each example's true <EOS>. Callers must use
    nn.CrossEntropyLoss(ignore_index=pad_idx) so padded positions don't
    contribute to the training signal.
    """

    def __init__(self, cfg: DecoderConfig, shared_embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.cfg = cfg

        if shared_embedding is not None:
            # Source and target are both DNA k-mer sequences over the identical
            # vocabulary, so tying the embedding matrix with the encoder's is a
            # deliberate parameter-reduction choice, not an oversight. Pass the
            # encoder's `nn.Embedding` instance in from
            # model/sequence_translation_model.py to enable this.
            assert shared_embedding.embedding_dim == cfg.embed_dim, (
                "shared_embedding dim must match DecoderConfig.embed_dim"
            )
            self.embedding = shared_embedding
        else:
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx)

        self.rle_embedder = RLEFeatureEmbedder(cfg.rle)
        self.attention = BahdanauAttention(cfg.attention)

        encoder_dim = cfg.attention.encoder_dim
        rle_dim = cfg.rle.output_dim

        cell_input_dim = cfg.embed_dim + encoder_dim + rle_dim
        self.decoder_cell = nn.LSTMCell(input_size=cell_input_dim, hidden_size=cfg.hidden_dim)
        self.hidden_dropout = nn.Dropout(cfg.dropout)

        # "Input feeding": the final vocabulary projection sees the raw hidden
        # state PLUS the context vectors directly, not just the hidden state
        # after they've been compressed through the recurrence. This gives the
        # output layer more direct access to "what the attention found" at
        # generation time.
        output_input_dim = cfg.hidden_dim + encoder_dim + rle_dim
        self.output_proj = nn.Sequential(
            nn.Linear(output_input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.vocab_size),
        )

    def forward(
        self,
        encoder_outputs: torch.Tensor,        # (B, L, encoder_dim)
        encoder_mask: torch.Tensor,           # (B, L) bool
        decoder_init_state,                   # tuple of (h0, c0), each (1, B, hidden_dim)
        rle_base_ids: torch.Tensor,           # (B, L) int64
        rle_run_lengths: torch.Tensor,        # (B, L) int64
        target_tokens: Optional[torch.Tensor] = None,   # (B, T_tgt) int64, includes leading <SOS>. None => pure inference.
        max_decode_len: int = 512,
        teacher_forcing_ratio: float = 1.0,
    ):
        B, L, _ = encoder_outputs.shape
        device = encoder_outputs.device

        h0, c0 = decoder_init_state
        h_t = h0.squeeze(0)  # (B, hidden_dim)
        c_t = c0.squeeze(0)  # (B, hidden_dim)

        rle_embeddings = self.rle_embedder(rle_base_ids, rle_run_lengths)  # (B, L, rle_dim)

        # Computed once per batch, reused at every decode step (see cross_attention.py docstring).
        encoder_proj = self.attention.project_encoder_outputs(encoder_outputs)  # (B, L, attn_dim)

        training_mode = target_tokens is not None
        if training_mode:
            num_steps = target_tokens.size(1) - 1  # predict tokens 1..T_tgt-1 from tokens 0..T_tgt-2
            if num_steps <= 0:
                raise ValueError(
                    "target_tokens must have length >= 2 (at least <SOS> plus one real token)."
                )
            prev_token = target_tokens[:, 0]  # should be <SOS> for every example
        else:
            num_steps = max_decode_len
            prev_token = torch.full((B,), self.cfg.sos_idx, dtype=torch.long, device=device)

        logits_steps = []
        predicted_steps = []
        alpha_steps = []

        # Tracks which sequences have already emitted <EOS>, for early-stopping
        # in pure inference mode. Kept purely as bookkeeping in training mode
        # (teacher forcing means we always run the full target length anyway).
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(num_steps):
            prev_embed = self.embedding(prev_token)  # (B, embed_dim)

            context, alpha = self.attention(h_t, encoder_outputs, encoder_proj, encoder_mask)
            # Reuse the SAME alpha to pool the RLE channel -- wherever the
            # decoder is attending in the standard encoder representation, it
            # also gets the homopolymer run-length signal from that same
            # position, rather than these two signals drifting independently.
            rle_context = torch.bmm(alpha.unsqueeze(1), rle_embeddings).squeeze(1)  # (B, rle_dim)

            cell_input = torch.cat([prev_embed, context, rle_context], dim=-1)
            h_t, c_t = self.decoder_cell(cell_input, (h_t, c_t))
            h_t_for_output = self.hidden_dropout(h_t)

            combined = torch.cat([h_t_for_output, context, rle_context], dim=-1)
            step_logits = self.output_proj(combined)  # (B, vocab_size)
            step_pred = step_logits.argmax(dim=-1)     # (B,)

            logits_steps.append(step_logits)
            predicted_steps.append(step_pred)
            alpha_steps.append(alpha)

            finished = finished | (step_pred == self.cfg.eos_idx)

            # Decide the NEXT step's input token.
            if training_mode:
                # Was: torch.rand(1, device=device).item() < teacher_forcing_ratio
                # That forced a CUDA device->host sync on every decode step (up to
                # chunk_size times per forward pass) for a plain Bernoulli coin flip
                # that never needed to touch the GPU at all. Especially costly under
                # WSL2, where each such round-trip crosses the driver boundary.
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    prev_token = target_tokens[:, t + 1]
                else:
                    prev_token = step_pred
            else:
                prev_token = step_pred
                if finished.all():
                    # Pure inference, whole batch has emitted <EOS>: no point
                    # decoding further padding steps. (Not done in training
                    # mode, where we must match target_tokens' fixed length.)
                    break

        logits = torch.stack(logits_steps, dim=1)              # (B, T', vocab_size)
        predicted_tokens = torch.stack(predicted_steps, dim=1)  # (B, T')
        attention_matrix = torch.stack(alpha_steps, dim=1)      # (B, T', L)  <-- XAI payload

        return logits, predicted_tokens, attention_matrix


# ---------------------------------------------------------------------------
# Integration sanity check: real HybridEncoder -> BahdanauAttention -> RLEDecoder,
# end to end, both in teacher-forcing (training) and free-running (inference) modes.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from model.hybrid_encoder import HybridEncoder, EncoderConfig
    from config import NUM_SPECIAL_TOKENS

    torch.manual_seed(0)

    B = 4
    L = 64  # source (encoder) length
    lengths = torch.tensor([64, 50, 33, 12], dtype=torch.int64)

    encoder_cfg = EncoderConfig()  # fused_dim=512 by default
    encoder = HybridEncoder(encoder_cfg)

    src_tokens = torch.full((B, L), PAD_IDX, dtype=torch.int64)
    for i, length in enumerate(lengths):
        src_tokens[i, :length] = torch.randint(
            NUM_SPECIAL_TOKENS, encoder_cfg.vocab_size, (int(length.item()),)
        )

    encoder_outputs, decoder_init_state, encoder_mask = encoder(src_tokens, lengths)
    print("encoder_outputs:", encoder_outputs.shape)

    # Fake RLE features aligned to the same L positions: random bases, random run lengths.
    rle_base_ids = torch.randint(0, RLE_BASE_VOCAB_SIZE - 1, (B, L))  # avoid PAD id in the "real" region
    rle_run_lengths = torch.randint(1, 15, (B, L))
    # zero out RLE features past each sequence's true length, mirroring encoder_mask
    for i, length in enumerate(lengths):
        rle_base_ids[i, length:] = RLE_PAD_BASE_IDX
        rle_run_lengths[i, length:] = 0

    decoder_cfg = DecoderConfig(
        attention=AttentionConfig(
            encoder_dim=encoder_cfg.fused_dim, decoder_dim=encoder_cfg.fused_dim, attn_dim=256
        )
    )
    decoder = RLEDecoder(decoder_cfg, shared_embedding=encoder.embedding)

    # --- Training-mode (teacher forcing) pass ---
    T_tgt = 20
    target_tokens = torch.randint(NUM_SPECIAL_TOKENS, decoder_cfg.vocab_size, (B, T_tgt))
    target_tokens[:, 0] = SOS_IDX
    target_tokens[:, -1] = EOS_IDX

    logits, predicted_tokens, attention_matrix = decoder(
        encoder_outputs=encoder_outputs,
        encoder_mask=encoder_mask,
        decoder_init_state=decoder_init_state,
        rle_base_ids=rle_base_ids,
        rle_run_lengths=rle_run_lengths,
        target_tokens=target_tokens,
        teacher_forcing_ratio=0.75,
    )

    print("\n[Training mode / teacher forcing]")
    print("logits:           ", logits.shape)              # (B, T_tgt - 1, vocab_size)
    print("predicted_tokens: ", predicted_tokens.shape)     # (B, T_tgt - 1)
    print("attention_matrix: ", attention_matrix.shape)     # (B, T_tgt - 1, L)

    assert logits.shape == (B, T_tgt - 1, decoder_cfg.vocab_size)
    assert predicted_tokens.shape == (B, T_tgt - 1)
    assert attention_matrix.shape == (B, T_tgt - 1, L)

    # Attention rows must sum to ~1 over valid positions (masking correctness check).
    row_sums = attention_matrix.sum(dim=-1)  # (B, T_tgt - 1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), "Attention rows must sum to 1"

    # Attention must not leak onto padded encoder positions.
    for i, length in enumerate(lengths):
        if length < L:
            leaked_mass = attention_matrix[i, :, length:].sum().item()
            assert leaked_mass < 1e-4, f"Attention leaked {leaked_mass} onto padding for example {i}"

    # --- Inference-mode (free-running / greedy) pass ---
    logits_inf, predicted_inf, attn_inf = decoder(
        encoder_outputs=encoder_outputs,
        encoder_mask=encoder_mask,
        decoder_init_state=decoder_init_state,
        rle_base_ids=rle_base_ids,
        rle_run_lengths=rle_run_lengths,
        target_tokens=None,
        max_decode_len=30,
    )
    print("\n[Inference mode / free-running]")
    print("logits:           ", logits_inf.shape)
    print("predicted_tokens: ", predicted_inf.shape)
    print("attention_matrix: ", attn_inf.shape)

    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal decoder parameters (excluding shared embedding): "
          f"{num_params - decoder.embedding.weight.numel():,}")
    print("\nIntegration sanity check passed.")