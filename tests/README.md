# tests/

Fast, pure-logic unit tests (no model forward passes, no Badread, no
minimap2 subprocess calls, no network) -- the whole suite runs in about
2 seconds and is safe to run on every commit.

Run with:
```
pytest tests/
```
(or `python -m pytest tests/` -- both work; imports resolve correctly
either way as long as you run from the project root)

## What's covered here vs. elsewhere

This directory covers the boundary logic that's easiest to get subtly
wrong and hardest to catch by inspection:

- `test_tokenizer.py` -- k-mer round-trip, homopolymer RLE correctness, and
  a permanent regression test for the boundary-padding bug (literal `N`
  padding silently forcing every sequence's edge tokens to `UNK`).
- `test_dataset.py` -- alignment-aware chunking: lossless reconstruction,
  boundary consistency, oversized-segment handling, malformed-breakpoint
  rejection.
- `test_inference_engine.py` -- chunk-stitching correctness, and a
  permanent regression test for the edlib query/target CIGAR labeling bug
  (insertions and deletions were briefly swapped during development).
- `test_metrics.py` -- the same edlib labeling check from a second call
  site, plus the frame-preservation nuance (a global "length matches mod 3"
  check can pass by coincidence while real local frameshifts occurred
  in between -- `frame_intact_fraction` is what catches that).

Slower, real-data integration tests intentionally live in each module's own
`if __name__ == "__main__":` block instead of here -- e.g.
`python -m data.simulator` runs real Badread against the real E. coli
genome, `python -m training.train` runs a real (small) training loop and
checks loss actually decreases, `python -m backend.main` spins up the real
FastAPI app via TestClient. Those prove the system works end-to-end against
real biological data; this directory proves specific logic is correct in
isolation, quickly. Both matter; neither is a substitute for the other.
