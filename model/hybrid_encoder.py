"""
hybrid_encoder.py
------------------
Hybrid Encoder for Context-Driven Neural Sequence Translation (ONT genome error correction).

Architecture:
    Embedding -> [Branch A: 1D-CNN motif detector]  ---\
                                                          >-- Fusion (concat + linear) --> encoder_outputs
              -> [Branch B: BiLSTM context reader]  ---/

Design note (read before wiring up the decoder/attention):
    The spec called for MaxPool1d-based downsampling in the CNN branch. That's the
    textbook CNN pattern, but it shrinks the temporal (sequence-length) dimension,
    while the BiLSTM branch preserves it. Bahdanau/Luong attention needs one encoder
    hidden vector per *input timestep* so the decoder can point back at a specific
    genomic position (that's also what the XAI heatmap is keyed on). If branch A and
    branch B disagree on L, you can't concatenate them per-timestep without an extra
    interpolation/alignment step.

    Fix used here: the CNN branch uses stride-1, same-padding convolutions, and
    pooling (if enabled) also uses stride=1 with same-padding, so it acts as local
    smoothing rather than downsampling. Sequence length L is preserved end-to-end.
    If you later want true downsampling for speed, you'll need to downsample BOTH
    branches identically (e.g. strided conv + strided LSTM readout) and update the
    attention/XAI coordinate mapping accordingly -- flagging this now so it doesn't
    surface as a silent shape-mismatch bug later.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import (
    K,
    NUM_KMERS,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    UNK_IDX,
    NUM_SPECIAL_TOKENS,
    VOCAB_SIZE,
)
# NOTE: these constants used to be defined locally in this file. They now
# live in config.py (the manifest's single, zero-dependency source of
# truth) so model/ and data/ can never silently disagree on what PAD_IDX,
# VOCAB_SIZE, etc. actually are. Re-exported here via the import above so
# any code still doing `from model.hybrid_encoder import PAD_IDX` continues
# to work unchanged.


@dataclass
class EncoderConfig:
    vocab_size: int = VOCAB_SIZE
    embed_dim: int = 256
    pad_idx: int = PAD_IDX

    # Branch A: CNN
    cnn_channels: tuple = (128, 128, 256)   # per-layer output channels
    cnn_kernel_sizes: tuple = (3, 5, 7)     # multi-scale receptive fields (motif widths vary)
    cnn_use_pooling: bool = True            # stride-1 smoothing pool, see module docstring
    cnn_dropout: float = 0.2

    # Branch B: BiLSTM
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3

    # Fusion
    fused_dim: int = 512
    fusion_dropout: float = 0.2


class MotifCNNBranch(nn.Module):
    """
    Branch A: 1D-CNN motif detector.

    Stacks Conv1d -> GELU -> (optional stride-1 MaxPool smoothing) layers with
    progressively larger channel counts, each using 'same' padding so the output
    sequence length always equals the input sequence length.

    Input:  x of shape (B, L, E)      [embedded tokens]
    Output: (B, L, C_out)             [C_out = cnn_channels[-1]]
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        assert len(cfg.cnn_channels) == len(cfg.cnn_kernel_sizes), (
            "cnn_channels and cnn_kernel_sizes must have matching lengths "
            "(one entry per conv layer)."
        )

        layers = []
        in_channels = cfg.embed_dim
        for out_channels, kernel_size in zip(cfg.cnn_channels, cfg.cnn_kernel_sizes):
            same_padding = kernel_size // 2
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=same_padding,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.GELU())
            if cfg.cnn_use_pooling:
                # stride=1 + same padding => smoothing, NOT downsampling (see module docstring)
                pool_k = 3
                layers.append(nn.MaxPool1d(kernel_size=pool_k, stride=1, padding=pool_k // 2))
            layers.append(nn.Dropout(cfg.cnn_dropout))
            in_channels = out_channels

        self.conv_stack = nn.Sequential(*layers)
        self.out_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, E) -> Conv1d expects (B, E, L)
        x = x.transpose(1, 2)
        out = self.conv_stack(x)  # (B, C_out, L)
        return out.transpose(1, 2)  # (B, L, C_out)


class ContextBiLSTMBranch(nn.Module):
    """
    Branch B: Bidirectional LSTM context reader.

    Reads the embedded sequence 5'->3' and 3'->5' in parallel, concatenating
    forward/backward hidden states at every timestep. Uses pack_padded_sequence
    so padding tokens are skipped rather than fed through the recurrence.

    Input:  x of shape (B, L, E), lengths of shape (B,) [true, unpadded lengths]
    Output: per-timestep states (B, L, 2*H), plus final (h_n, c_n) for decoder init
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.lstm_dropout if cfg.lstm_num_layers > 1 else 0.0,
        )
        self.hidden_dim = cfg.lstm_hidden_dim
        self.num_layers = cfg.lstm_num_layers

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # lengths must be on CPU, int64, for pack_padded_sequence
        lengths_cpu = lengths.detach().to("cpu", dtype=torch.int64)

        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        # outputs: (B, L, 2*H) -- forward/backward already concatenated per timestep
        return outputs, (h_n, c_n)


class HybridEncoder(nn.Module):
    """
    Full hybrid encoder: shared embedding -> CNN branch + BiLSTM branch -> fusion.

    forward() returns:
        encoder_outputs : (B, L, fused_dim)
            Per-timestep fused representation. This is what the cross-attention
            layer in the decoder will attend over to produce alpha_{t,i}.
        decoder_init_state : (h_0, c_0), each (1, B, fused_dim)
            A single-layer, unidirectional initial state for the LSTMCell/GRUCell
            decoder, derived by projecting the BiLSTM's final forward+backward
            states down to fused_dim.
        mask : (B, L) bool
            True at real-token positions, False at padding. Pass this into the
            attention softmax (as an additive -inf mask) so the decoder can never
            attend to padding -- otherwise attention mass can leak onto <PAD>
            positions, which is a common silent bug in seq2seq implementations.
    """

    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.embed_dim,
            padding_idx=cfg.pad_idx,
        )

        self.cnn_branch = MotifCNNBranch(cfg)
        self.bilstm_branch = ContextBiLSTMBranch(cfg)

        cnn_out_dim = self.cnn_branch.out_channels
        lstm_out_dim = cfg.lstm_hidden_dim * 2  # bidirectional

        self.fusion_proj = nn.Sequential(
            nn.Linear(cnn_out_dim + lstm_out_dim, cfg.fused_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.fused_dim),
            nn.Dropout(cfg.fusion_dropout),
        )

        # Project BiLSTM's final (num_layers * 2, B, H) state down to a single
        # (1, B, fused_dim) init state for the unidirectional decoder cell.
        self.decoder_init_proj_h = nn.Linear(lstm_out_dim, cfg.fused_dim)
        self.decoder_init_proj_c = nn.Linear(lstm_out_dim, cfg.fused_dim)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor):
        """
        token_ids : (B, L) int64, k-mer/token indices (already includes any
                    padding, SOS/EOS handling done upstream in the Dataset).
        lengths   : (B,)  int64, true sequence length per example (<= L).
        """
        B, L = token_ids.shape

        embedded = self.embedding(token_ids)  # (B, L, E)

        cnn_out = self.cnn_branch(embedded)  # (B, L, C_cnn)
        lstm_out, (h_n, c_n) = self.bilstm_branch(embedded, lengths)  # (B, L, 2H)

        fused_input = torch.cat([cnn_out, lstm_out], dim=-1)  # (B, L, C_cnn + 2H)
        encoder_outputs = self.fusion_proj(fused_input)  # (B, L, fused_dim)

        # Build final bidirectional state from the LAST LSTM layer's
        # forward/backward directions: h_n shape is (num_layers*2, B, H).
        last_layer_fwd = h_n[-2]  # (B, H)
        last_layer_bwd = h_n[-1]  # (B, H)
        last_c_fwd = c_n[-2]
        last_c_bwd = c_n[-1]

        final_h_bidir = torch.cat([last_layer_fwd, last_layer_bwd], dim=-1)  # (B, 2H)
        final_c_bidir = torch.cat([last_c_fwd, last_c_bwd], dim=-1)          # (B, 2H)

        decoder_h0 = self.decoder_init_proj_h(final_h_bidir).unsqueeze(0)  # (1, B, fused_dim)
        decoder_c0 = self.decoder_init_proj_c(final_c_bidir).unsqueeze(0)  # (1, B, fused_dim)

        # Padding mask for attention (True = attend-able position)
        position_ids = torch.arange(L, device=token_ids.device).unsqueeze(0)  # (1, L)
        mask = position_ids < lengths.unsqueeze(1)  # (B, L) bool

        return encoder_outputs, (decoder_h0, decoder_c0), mask


# ---------------------------------------------------------------------------
# Sanity check: run a fake batch through the encoder and verify shapes.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = EncoderConfig()
    encoder = HybridEncoder(cfg)

    batch_size = 4
    max_len = 64
    lengths = torch.tensor([64, 50, 33, 12], dtype=torch.int64)

    # Random token ids, respecting each example's true length; pad the rest with PAD_IDX.
    token_ids = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.int64)
    for i, length in enumerate(lengths):
        token_ids[i, :length] = torch.randint(NUM_SPECIAL_TOKENS, cfg.vocab_size, (int(length.item()),))

    encoder_outputs, (h0, c0), mask = encoder(token_ids, lengths)
    
    print("encoder_outputs:", encoder_outputs.shape)   # expect (4, 64, fused_dim)
    print("decoder h0:     ", h0.shape)                # expect (1, 4, fused_dim)
    print("decoder c0:     ", c0.shape)                # expect (1, 4, fused_dim)
    print("mask:           ", mask.shape, mask.dtype)  # expect (4, 64) bool
    print("mask row 3 (len=12):", mask[3].int().tolist())

    assert encoder_outputs.shape == (batch_size, max_len, cfg.fused_dim)
    assert h0.shape == (1, batch_size, cfg.fused_dim)
    assert c0.shape == (1, batch_size, cfg.fused_dim)
    assert mask.shape == (batch_size, max_len)
    assert mask[3].sum().item() == 12  # only first 12 positions are real tokens

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal encoder parameters: {num_params:,}")
    print("Sanity check passed.")
