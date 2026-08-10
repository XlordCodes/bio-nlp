"""
sequence_translation_model.py
-------------------------------
Top-level system class: wires HybridEncoder + RLEDecoder into a single
end-to-end PyTorch module with one clean forward() contract.

Responsibilities that live HERE and nowhere else:
    1. Own the two sub-config objects and validate that their shared
       dimensions actually agree (fused_dim / hidden_dim / attention dims),
       so a mismatch is caught at model-construction time with a clear
       message -- not 40 stack frames deep inside an LSTMCell shape error.
    2. Tie the encoder's token embedding into the decoder, satisfying the
       shared-vocabulary constraint (source k-mers and target k-mers are the
       same 4096+4 token space).
    3. Return the decoder's attention matrix completely unmodified, so
       backend/inference_engine.py can serialize exactly what the network
       actually computed -- no truncation, no averaging, no resolution
       changes happen in this file.

This file does NOT load checkpoints from disk (that's
backend/inference_engine.py's job) and does NOT compute the loss (that's
training/train.py's job). It only defines the model's forward computation.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from model.hybrid_encoder import HybridEncoder, EncoderConfig
from model.rle_decoder import RLEDecoder, DecoderConfig


@dataclass
class SequenceTranslationConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    tie_embeddings: bool = True  # source/target k-mer vocab is identical -- tie by default


@dataclass
class TranslationOutput:
    """
    Structured forward()/predict() output. Using a dataclass instead of a bare
    tuple or dict so downstream call sites (training loop, inference engine)
    get named, typo-proof field access: output.attention_matrix, not
    output["attn_matrx"] silently returning a KeyError three modules away.
    """

    logits: torch.Tensor            # (B, T, vocab_size) -- raw scores; feed to CrossEntropyLoss
    predicted_tokens: torch.Tensor  # (B, T)              -- argmax token ids at each decode step
    attention_matrix: torch.Tensor  # (B, T, L)           -- unmodified alpha, straight from the decoder
    encoder_mask: torch.Tensor      # (B, L) bool         -- which source positions were real vs <PAD>


class SequenceTranslationModel(nn.Module):
    """
    Full encoder-decoder system.

    forward(): used during TRAINING. Pass `target_tokens` (with leading <SOS>)
               to run teacher forcing at the given ratio. Gradients flow
               normally; caller is responsible for the optimizer step and loss.

    predict(): convenience wrapper for INFERENCE. Puts the model in eval mode,
               wraps execution in torch.no_grad() (per the Part 4 backend
               spec), and runs free-running (greedy) decoding since no target
               is available. This is what backend/inference_engine.py should
               call directly.
    """

    def __init__(self, cfg: SequenceTranslationConfig = SequenceTranslationConfig()):
        super().__init__()
        self._validate_config(cfg)
        self.cfg = cfg

        self.encoder = HybridEncoder(cfg.encoder)

        shared_embedding = self.encoder.embedding if cfg.tie_embeddings else None
        self.decoder = RLEDecoder(cfg.decoder, shared_embedding=shared_embedding)

    @staticmethod
    def _validate_config(cfg: SequenceTranslationConfig) -> None:
        enc, dec = cfg.encoder, cfg.decoder

        if enc.fused_dim != dec.hidden_dim:
            raise ValueError(
                f"Config mismatch: encoder.fused_dim ({enc.fused_dim}) must equal "
                f"decoder.hidden_dim ({dec.hidden_dim}). The encoder's final projection "
                f"and the decoder LSTMCell's initial state size must match, since "
                f"HybridEncoder.forward() hands (h0, c0) of shape (1, B, fused_dim) "
                f"directly to the decoder as its starting state."
            )

        if enc.fused_dim != dec.attention.encoder_dim:
            raise ValueError(
                f"Config mismatch: encoder.fused_dim ({enc.fused_dim}) must equal "
                f"decoder.attention.encoder_dim ({dec.attention.encoder_dim}). The "
                f"attention module's W_enc projects encoder_outputs, whose last dim "
                f"is fused_dim."
            )

        if dec.hidden_dim != dec.attention.decoder_dim:
            raise ValueError(
                f"Config mismatch: decoder.hidden_dim ({dec.hidden_dim}) must equal "
                f"decoder.attention.decoder_dim ({dec.attention.decoder_dim}). The "
                f"attention module's W_dec projects the decoder LSTMCell's hidden "
                f"state, so these must agree."
            )

        if cfg.tie_embeddings:
            if enc.vocab_size != dec.vocab_size:
                raise ValueError(
                    f"Config mismatch: tie_embeddings=True but encoder.vocab_size "
                    f"({enc.vocab_size}) != decoder.vocab_size ({dec.vocab_size}). "
                    f"Source and target must share the exact same vocabulary to tie "
                    f"their embedding tables."
                )
            if enc.embed_dim != dec.embed_dim:
                raise ValueError(
                    f"Config mismatch: tie_embeddings=True but encoder.embed_dim "
                    f"({enc.embed_dim}) != decoder.embed_dim ({dec.embed_dim}). A "
                    f"single nn.Embedding instance is being shared between encoder "
                    f"and decoder, so its dimensionality must be identical on both "
                    f"sides -- there is no separate projection layer to reconcile a "
                    f"mismatch."
                )

    def forward(
        self,
        src_tokens: torch.Tensor,        # (B, L) int64
        src_lengths: torch.Tensor,       # (B,)   int64, true (unpadded) source lengths
        rle_base_ids: torch.Tensor,      # (B, L) int64, aligned to src_tokens positions
        rle_run_lengths: torch.Tensor,   # (B, L) int64, aligned to src_tokens positions
        target_tokens: Optional[torch.Tensor] = None,  # (B, T_tgt) int64, includes leading <SOS>. None => free-running.
        max_decode_len: int = 512,
        teacher_forcing_ratio: float = 1.0,
    ) -> TranslationOutput:
        encoder_outputs, decoder_init_state, encoder_mask = self.encoder(src_tokens, src_lengths)

        logits, predicted_tokens, attention_matrix = self.decoder(
            encoder_outputs=encoder_outputs,
            encoder_mask=encoder_mask,
            decoder_init_state=decoder_init_state,
            rle_base_ids=rle_base_ids,
            rle_run_lengths=rle_run_lengths,
            target_tokens=target_tokens,
            max_decode_len=max_decode_len,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        return TranslationOutput(
            logits=logits,
            predicted_tokens=predicted_tokens,
            attention_matrix=attention_matrix,  # returned exactly as the decoder produced it
            encoder_mask=encoder_mask,
        )

    @torch.no_grad()
    def predict(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        rle_base_ids: torch.Tensor,
        rle_run_lengths: torch.Tensor,
        max_decode_len: int = 512,
    ) -> TranslationOutput:
        """
        Inference-only entry point for backend/inference_engine.py.

        Deactivates autograd (torch.no_grad()) per the Part 4 spec, to avoid
        allocating gradient-tracking memory during serving, and forces the
        module into eval() mode so dropout and any batch-norm running-stats
        behave correctly at inference time. Restores the model's previous
        training/eval mode afterward so calling predict() mid-training (e.g.
        during a validation loop) doesn't accidentally leave the model in
        eval mode for the next training step.
        """
        was_training = self.training
        self.eval()
        try:
            return self.forward(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                rle_base_ids=rle_base_ids,
                rle_run_lengths=rle_run_lengths,
                target_tokens=None,
                max_decode_len=max_decode_len,
                teacher_forcing_ratio=0.0,
            )
        finally:
            self.train(was_training)

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# End-to-end sanity check: build the full tied model, run both a training-mode
# (teacher forcing) forward pass and an inference-mode predict() call, and
# verify the embedding tying actually holds (same underlying storage, not just
# same shape).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import PAD_IDX, SOS_IDX, EOS_IDX, NUM_SPECIAL_TOKENS, RLE_BASE_VOCAB_SIZE

    torch.manual_seed(0)

    cfg = SequenceTranslationConfig()  # all defaults: fused_dim=512, hidden_dim=512, tie_embeddings=True
    model = SequenceTranslationModel(cfg)

    # --- Verify embeddings are genuinely tied (same tensor storage) ---
    assert model.encoder.embedding.weight.data_ptr() == model.decoder.embedding.weight.data_ptr(), (
        "Embeddings should be the exact same parameter tensor when tie_embeddings=True"
    )
    print("Embedding tying verified: encoder and decoder share the same weight storage.")

    B, L = 4, 64
    lengths = torch.tensor([64, 50, 33, 12], dtype=torch.int64)

    src_tokens = torch.full((B, L), PAD_IDX, dtype=torch.int64)
    for i, length in enumerate(lengths):
        src_tokens[i, :length] = torch.randint(NUM_SPECIAL_TOKENS, cfg.encoder.vocab_size, (int(length.item()),))
        
    rle_base_ids = torch.randint(0, RLE_BASE_VOCAB_SIZE - 1, (B, L))
    rle_run_lengths = torch.randint(1, 15, (B, L))
    for i, length in enumerate(lengths):
        rle_base_ids[i, length:] = RLE_BASE_VOCAB_SIZE - 1  # PAD base id
        rle_run_lengths[i, length:] = 0

    # --- Training-mode forward pass (teacher forcing), gradients enabled ---
    T_tgt = 20
    target_tokens = torch.randint(NUM_SPECIAL_TOKENS, cfg.decoder.vocab_size, (B, T_tgt))
    target_tokens[:, 0] = SOS_IDX
    target_tokens[:, -1] = EOS_IDX

    output = model(
        src_tokens=src_tokens,
        src_lengths=lengths,
        rle_base_ids=rle_base_ids,
        rle_run_lengths=rle_run_lengths,
        target_tokens=target_tokens,
        teacher_forcing_ratio=0.75,
    )

    print("\n[Training mode]")
    print("logits:           ", output.logits.shape)
    print("predicted_tokens: ", output.predicted_tokens.shape)
    print("attention_matrix: ", output.attention_matrix.shape)
    print("encoder_mask:     ", output.encoder_mask.shape)

    assert output.logits.shape == (B, T_tgt - 1, cfg.decoder.vocab_size)
    assert output.attention_matrix.shape == (B, T_tgt - 1, L)

    # Confirm gradients actually flow end-to-end through both branches of the
    # encoder and through the tied embedding.
    loss = output.logits.sum()
    loss.backward()
    assert model.encoder.embedding.weight.grad is not None, "Gradient must reach the tied embedding"
    assert model.encoder.cnn_branch.conv_stack[0].weight.grad is not None, "Gradient must reach the CNN branch"
    assert model.encoder.bilstm_branch.lstm.weight_ih_l0.grad is not None, "Gradient must reach the BiLSTM branch"
    print("\nBackward pass verified: gradients reach the tied embedding, CNN branch, and BiLSTM branch.")

    # --- Inference-mode predict() call: no target, no grad, eval mode ---
    model.zero_grad()
    inference_output = model.predict(
        src_tokens=src_tokens,
        src_lengths=lengths,
        rle_base_ids=rle_base_ids,
        rle_run_lengths=rle_run_lengths,
        max_decode_len=30,
    )
    print("\n[Inference mode / predict()]")
    print("logits:           ", inference_output.logits.shape)
    print("predicted_tokens: ", inference_output.predicted_tokens.shape)
    print("attention_matrix: ", inference_output.attention_matrix.shape)
    assert inference_output.logits.requires_grad is False, "predict() must not build an autograd graph"

    # --- Also verify tie_embeddings=False produces genuinely separate tables ---
    cfg_untied = SequenceTranslationConfig(tie_embeddings=False)
    model_untied = SequenceTranslationModel(cfg_untied)
    assert model_untied.encoder.embedding.weight.data_ptr() != model_untied.decoder.embedding.weight.data_ptr(), (
        "Embeddings must be separate parameter tensors when tie_embeddings=False"
    )
    print("\ntie_embeddings=False path verified: encoder and decoder have independent embedding tables.")

    print(f"\nTotal trainable parameters (tied model): {model.count_parameters():,}")
    print("\nFull end-to-end sanity check passed.")
