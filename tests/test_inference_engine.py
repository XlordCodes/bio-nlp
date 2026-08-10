"""
tests/test_inference_engine.py
---------------------------------
Fast, pure-logic unit tests for backend/inference_engine.py's stitching and
metrics functions. Uses fabricated corrected-chunk strings directly (no
model forward pass) so these run in milliseconds -- see
backend/inference_engine.py's own __main__ block for the full model +
real-genome integration proof.
"""

import pytest

from backend.inference_engine import InferenceEngine


@pytest.fixture(scope="module")
def engine():
    """A model is required by InferenceEngine.__init__ but never actually
    runs a forward pass in these tests -- only _stitch_chunks/_compute_metrics
    (both @staticmethod-adjacent, called directly) are exercised. Module-scoped
    since these tests never mutate the model, and construction alone (no
    forward pass) is what's slow here, not the stitching/metrics logic itself."""
    import torch
    from model.sequence_translation_model import SequenceTranslationConfig, SequenceTranslationModel

    model = SequenceTranslationModel(SequenceTranslationConfig())
    return InferenceEngine(model, chunk_size=100, chunk_overlap=20, device=torch.device("cpu"))


def test_edlib_insertion_deletion_labeling_against_ground_truth():
    """
    Regression test for a real bug caught during development: edlib's CIGAR
    'I'/'D' ops are relative to query-vs-target, not "input-vs-output"
    intuition. query=input, target=corrected here, so a base present ONLY
    in the corrected output is an INSERTION (edlib op 'D'), and a base
    present ONLY in the input is a DELETION (edlib op 'I') -- backwards
    from what the letters suggest at a glance. This test locks in the
    corrected labeling permanently.
    """
    reference = "ACGTACGTACGT"

    corrected_with_extra_base = "ACGTAACGTACGT"  # corrected is LONGER -> the model inserted a base
    metrics = InferenceEngine._compute_metrics(reference, corrected_with_extra_base, num_chunks=1, latency_ms=1.0)
    assert metrics["num_insertions"] == 1
    assert metrics["num_deletions"] == 0

    corrected_missing_base = "ACGTACGTACG"  # corrected is SHORTER -> the model deleted a base
    metrics = InferenceEngine._compute_metrics(reference, corrected_missing_base, num_chunks=1, latency_ms=1.0)
    assert metrics["num_deletions"] == 1
    assert metrics["num_insertions"] == 0


def test_metrics_identical_sequences_have_zero_edit_distance():
    metrics = InferenceEngine._compute_metrics("ACGTACGT", "ACGTACGT", num_chunks=1, latency_ms=1.0)
    assert metrics["edit_distance"] == 0
    assert metrics["num_matches"] == 8


def test_stitching_single_chunk_passes_through_unchanged(engine):
    final_seq, offsets = engine._stitch_chunks([(0, 50)], ["ACGTACGTAC"])
    assert final_seq == "ACGTACGTAC"
    assert offsets == [(0, 10)]


def test_stitching_two_chunks_produces_contiguous_output(engine):
    """Fabricated two-chunk case: both chunks 'corrected' the same overlap
    region identically, so stitching should produce a single coherent
    sequence with no duplicated or missing bases at the seam."""
    chunk_offsets = [(0, 60), (40, 100)]  # 20-base overlap in source coordinates
    corrected_a = "A" * 40 + "CGTACGTACGTACGTACGT"  # 60 chars, last 20 = the "overlap"
    corrected_b = "CGTACGTACGTACGTACGT" + "T" * 40  # 60 chars, first 20 = the "overlap" (identical content)

    final_seq, offsets = engine._stitch_chunks(chunk_offsets, [corrected_a, corrected_b])

    # Contiguous, non-overlapping contributions covering the whole output.
    assert offsets[0][0] == 0
    assert offsets[0][1] == offsets[1][0]
    assert offsets[1][1] == len(final_seq)
    # No characters invented or dropped beyond what stitching intentionally trims.
    assert final_seq.startswith("A" * 40)
    assert final_seq.endswith("T" * 40)


def test_stitching_handles_empty_chunk_without_crashing(engine):
    """An untrained/degenerate model could emit an empty corrected chunk
    (EOS immediately). Stitching must not crash on this."""
    final_seq, offsets = engine._stitch_chunks([(0, 50), (30, 80)], ["", "ACGTACGT"])
    assert isinstance(final_seq, str)
    assert len(offsets) == 2
