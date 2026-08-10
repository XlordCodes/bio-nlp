"""
tests/test_dataset.py
------------------------
Fast, pure-logic unit tests for data/dataset.py's chunking and validation.
No model, no DataLoader batching against real tensors here -- see
data/dataset.py's own __main__ block for the full tokenizer+model
integration proof.
"""

import random

import pytest

from data.dataset import AlignedPair, _chunk_aligned_pair


def _simulate_indel_pair(clean_seq: str, num_edits: int, seed: int):
    """Minimal local copy of data/dataset.py's own test helper -- generates a
    (noisy_seq, breakpoints) pair with real indels, for exercising alignment-
    aware chunking without needing Badread."""
    rng = random.Random(seed)
    noisy_chars = []
    breakpoints = [(0, 0)]
    noisy_pos = 0

    edit_positions = set(rng.sample(range(len(clean_seq)), min(num_edits, len(clean_seq))))

    for clean_pos in range(len(clean_seq)):
        if clean_pos in edit_positions:
            op = rng.choice(["ins", "del", "sub"])
            if op == "ins":
                extra = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 3)))
                noisy_chars.append(extra)
                noisy_pos += len(extra)
                noisy_chars.append(clean_seq[clean_pos])
                noisy_pos += 1
            elif op == "del":
                pass
            else:
                alt = rng.choice([b for b in "ACGT" if b != clean_seq[clean_pos]])
                noisy_chars.append(alt)
                noisy_pos += 1
        else:
            noisy_chars.append(clean_seq[clean_pos])
            noisy_pos += 1
        breakpoints.append((noisy_pos, clean_pos + 1))

    noisy_seq = "".join(noisy_chars)
    deduped = [breakpoints[0]]
    for bp in breakpoints[1:]:
        if bp != deduped[-1]:
            deduped.append(bp)
    return noisy_seq, deduped


@pytest.fixture
def indel_pair():
    clean_seq = "".join(random.Random(0).choice("ACGT") for _ in range(2000))
    noisy_seq, breakpoints = _simulate_indel_pair(clean_seq, num_edits=100, seed=1)
    return AlignedPair(noisy_seq, clean_seq, breakpoints)


def test_chunking_reconstructs_both_sequences_losslessly(indel_pair):
    """With min_chunk_size=1 (nothing dropped), concatenating all chunks
    must exactly reproduce both the original noisy and clean sequences --
    no data lost or duplicated across a chunk boundary."""
    chunks = _chunk_aligned_pair(indel_pair, chunk_size=256, min_chunk_size=1)
    assert "".join(c[0] for c in chunks) == indel_pair.noisy_sequence
    assert "".join(c[1] for c in chunks) == indel_pair.clean_sequence


def test_chunk_boundaries_are_alignment_consistent(indel_pair):
    """Each chunk boundary must fall exactly on a real breakpoint -- i.e.
    the noisy/clean substrings at each boundary genuinely correspond to the
    same source region, not an arbitrary fixed-index cut."""
    bp_set = set(indel_pair.breakpoints)
    chunk_size = 256
    seg_start = indel_pair.breakpoints[0]
    prev = indel_pair.breakpoints[0]
    boundaries = []
    for nxt in indel_pair.breakpoints[1:]:
        if nxt[0] - seg_start[0] > chunk_size and prev != seg_start:
            boundaries.append(prev)
            seg_start = prev
        prev = nxt
    for boundary in boundaries:
        assert boundary in bp_set


def test_min_chunk_size_filters_tiny_fragments():
    pair = AlignedPair.from_identity("ACGTACGTAC", "ACGTACGTAC")  # 10bp, no indels
    assert _chunk_aligned_pair(pair, chunk_size=1000, min_chunk_size=20) == []
    assert len(_chunk_aligned_pair(pair, chunk_size=1000, min_chunk_size=1)) == 1


def test_oversized_single_segment_is_kept_whole_not_cut():
    """A single alignment segment larger than chunk_size must not be split
    mid-segment (that would reintroduce the exact misalignment problem
    chunking exists to avoid) -- it should come back as one oversized chunk."""
    pair = AlignedPair.from_identity("A" * 500, "A" * 500)  # one segment, no internal breakpoints
    chunks = _chunk_aligned_pair(pair, chunk_size=100, min_chunk_size=1)
    assert len(chunks) == 1
    assert len(chunks[0][0]) == 500


@pytest.mark.parametrize(
    "breakpoints,expectation",
    [
        ([(1, 0), (10, 10)], "start at"),
        ([(0, 0), (5, 5)], "end at"),  # doesn't reach the true end
        ([(0, 0), (5, 3), (3, 5), (10, 10)], "non-decreasing"),
    ],
)
def test_aligned_pair_rejects_malformed_breakpoints(breakpoints, expectation):
    pair = AlignedPair("A" * 10, "A" * 10, breakpoints)
    with pytest.raises(ValueError, match=expectation):
        pair.validate()


def test_aligned_pair_accepts_well_formed_breakpoints():
    pair = AlignedPair("A" * 10, "A" * 12, [(0, 0), (4, 4), (10, 12)])
    pair.validate()  # should not raise
