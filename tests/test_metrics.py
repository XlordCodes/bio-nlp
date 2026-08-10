"""
tests/test_metrics.py
------------------------
Fast, pure-logic unit tests for training/metrics.py. No minimap2 subprocess
calls here (that's exercised in training/metrics.py's own __main__ block,
against a real genome excerpt) -- these test the edlib-based functions only.
"""

from training.metrics import compute_alignment_metrics, compute_frame_preservation


def test_identical_sequences_score_perfectly():
    seq = "ACGTACGTACGTGGCCTTAACGTACGT"
    m = compute_alignment_metrics(seq, seq)
    assert m["edit_distance"] == 0
    assert m["identity"] == 1.0

    f = compute_frame_preservation(seq, seq)
    assert f["global_frame_preserved"] is True
    assert f["frame_intact_fraction"] == 1.0
    assert f["num_frameshift_events"] == 0


def test_insertion_deletion_labeling_against_ground_truth():
    """Same edlib query/target convention risk as backend/inference_engine.py
    -- re-verified independently here since it's a separate call site."""
    ref = "ACGTACGTACGT"

    pred_with_extra_base = "ACGTAACGTACGT"
    m = compute_alignment_metrics(pred_with_extra_base, ref)
    assert m["num_insertions_remaining"] == 1
    assert m["num_deletions_remaining"] == 0

    pred_missing_base = "ACGTACGTACG"
    m = compute_alignment_metrics(pred_missing_base, ref)
    assert m["num_deletions_remaining"] == 1
    assert m["num_insertions_remaining"] == 0


def test_single_uncorrected_deletion_breaks_frame_for_the_rest():
    reference = "ATGGCTAAACGTGGGCCCTTTAAACGTGGGCCCTTTAAA"
    single_del = reference[:15] + reference[16:]
    f = compute_frame_preservation(single_del, reference)
    assert f["global_frame_preserved"] is False
    assert f["frame_intact_fraction"] < 1.0
    assert f["num_frameshift_events"] == 1


def test_global_frame_check_can_be_misleadingly_satisfied():
    """
    Three separate uncorrected single-base deletions can cancel out to a
    net offset divisible by 3, making the GLOBAL frame-preserved check pass
    even though real errors occurred and the regions between them are
    locally out of frame. frame_intact_fraction must reveal this even when
    the global boolean doesn't -- this is the whole reason both metrics are
    reported rather than just the global one.
    """
    reference = "ATGGCTAAACGTGGGCCCTTTAAACGTGGGCCCTTTAAA"
    triple_del = reference
    for pos in sorted([10, 20, 30], reverse=True):
        triple_del = triple_del[:pos] + triple_del[pos + 1 :]

    f = compute_frame_preservation(triple_del, reference)
    assert f["global_frame_preserved"] is True  # net -3 offset, divisible by 3
    assert f["frame_intact_fraction"] < 1.0  # but real local corruption occurred
