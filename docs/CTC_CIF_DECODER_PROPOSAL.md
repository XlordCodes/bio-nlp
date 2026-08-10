# Proposal: Non-Autoregressive Decoding (Stage 2/3)

**Status: proposal, not implemented.** This touches `model/rle_decoder.py`,
`model/cross_attention.py`, `model/sequence_translation_model.py`, and
`backend/inference_engine.py`'s stitching/max-decode-length logic -- most of
the model layer. Given the scope, this is a decision point, not something to
build unprompted.

## The problem it solves

Research confirms the concern directly: our autoregressive decoder generates
one token at a time in a sequential Python loop. For 10-50kb reads, this is
the single biggest latency risk to real-world viability -- "if a model takes
longer to correct a read than it took to physically sequence it, it cannot
be deployed in production." BaseNet (`github.com/liqingwen98/BaseNet`, MIT
license, 2024) is real, existing prior art for exactly this problem in
nanopore sequence decoding, combining CTC and CIF/Paraformer approaches.

## Two candidate approaches, and why CIF fits us better than CTC

**CTC (Connectionist Temporal Classification)** outputs a token distribution
at every encoder position, with a "blank" token allowing collapse/removal
during decoding. It's the more common NAR choice in ASR/basecalling.
**Problem for us specifically**: CTC's alignment is monotonic and can only
shrink or preserve length via blank/repeat collapsing -- it cannot natively
produce an output *longer* than the encoder's sequence length. Our whole
architecture exists because indel correction requires the decoder to
sometimes *insert* bases the noisy input never had. Naive CTC would silently
break exactly the capability that motivated choosing Seq2Seq over
classification in the first place (see Part 1 of the original design brief).
A CTC approach would need an explicit upsampling/duration-prediction module
before the CTC head to create "room" for insertions -- a real, added
component, not a drop-in swap.

**CIF (Continuous Integrate-and-Fire)** accumulates a weighted sum of
encoder states over time and "fires" an output token whenever accumulated
weight crosses a threshold. Critically, **the number of firings is learned
and dynamic**, not tied to input length -- this handles both insertions and
deletions relative to the input naturally, without a separate upsampling
component. **Recommendation: CIF over raw CTC**, specifically because of
this length-flexibility match to our actual task.

## Sketch of the redesign (for discussion, not final)

- Replace the autoregressive `LSTMCell` loop in `rle_decoder.py` with a CIF
  integrate-and-fire module sitting on top of the existing BiLSTM encoder
  output -- the encoder itself (CNN + BiLSTM + fusion) doesn't need to
  change.
- The RLE channel's fusion-via-shared-attention-weights mechanism needs
  rethinking: CIF's firing weights would take over the role Bahdanau
  attention's alpha currently plays, so the RLE context vector should be
  pooled using CIF's firing weights instead. This preserves the design
  intent (RLE and sequence context fused through the same weighting) with
  a different underlying weighting mechanism.
- The XAI attention-matrix export (the whole point of `cross_attention.py`
  returning alpha explicitly) needs a CIF-native equivalent -- firing
  weights over encoder positions are a reasonable analog, but this is worth
  making an explicit decision about, not an assumption.
- `backend/inference_engine.py`'s `max_decode_len` budget and
  `EOS_IDX`-based stopping logic become unnecessary (CIF's firing count IS
  the output length) -- this actually *simplifies* the inference engine in
  some ways.
- Following BaseNet's own pattern, a **joint-loss training setup** (forward
  + reverse decoders during training only) is worth adopting regardless of
  CTC vs. CIF -- it's reported to meaningfully improve convergence and
  doesn't change the deployed model's architecture at all.

## What I'd want confirmed before building this

1. Whether the added engineering complexity is worth it *before* Stage 0
   produces a single real accuracy number on the current architecture --
   redesigning the decoder before knowing whether the current one is even
   accuracy-competitive risks optimizing the wrong thing first.
2. Whether you want a full CIF rewrite, or a smaller intermediate step
   first (e.g., just the joint forward/reverse loss addition, which is
   compatible with the current autoregressive decoder and much lower risk).

## Recommendation

Sequence this **after** Stage 0's real training run, not before. If the
current autoregressive architecture already clears a competitive Q-score,
the CIF rewrite becomes a pure performance project with a known accuracy
target to preserve. If it doesn't, you'd want to know that before investing
in a decoder rewrite that inherits the same underlying accuracy ceiling.
