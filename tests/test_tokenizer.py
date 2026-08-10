"""
tests/test_tokenizer.py
--------------------------
Fast, pure-logic unit tests for data/tokenizer.py. No model, no Badread, no
network -- these should run in well under a second and are safe to run on
every commit. (Slower, real-data integration checks live in each module's
own `if __name__ == "__main__":` block instead -- see tests/README.md.)
"""

import pytest

from config import RLE_BASE_TO_IDX, UNK_IDX, LEFT_PAD_BASES, RIGHT_PAD_BASES
from data.tokenizer import KmerTokenizer


@pytest.fixture
def tokenizer():
    return KmerTokenizer()


@pytest.mark.parametrize("length", [1, 5, 6, 7, 50, 200])
def test_encode_decode_round_trip(tokenizer, length):
    import random

    rng = random.Random(length)  # deterministic per length
    seq = "".join(rng.choice("ACGT") for _ in range(length))
    tokenized = tokenizer.encode(seq)
    assert len(tokenized.token_ids) == length
    assert tokenizer.decode(tokenized.token_ids) == seq


def test_pure_acgt_sequence_has_no_unk_tokens(tokenizer):
    """
    Regression test for a real bug caught during development: padding
    boundary windows with a literal 'N' character made the first
    LEFT_PAD_BASES and last RIGHT_PAD_BASES tokens of every sequence UNK,
    even for sequences with no actual ambiguity. Fixed via edge-replicate
    padding. This test exists specifically so that fix can never silently
    regress.
    """
    seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    tokenized = tokenizer.encode(seq)
    assert UNK_IDX not in tokenized.token_ids, (
        "A pure-ACGT sequence produced UNK tokens -- boundary padding is likely "
        "using a literal ambiguous character again instead of edge-replication."
    )


def test_homopolymer_run_length_is_correct(tokenizer):
    seq = "GATTACAAAAATCGGGGGCTA"
    #  index: 0:G 1:A 2:T 3:T 4:A 5:C 6:A 7:A 8:A 9:A 10:A 11:T 12:C
    #         13:G 14:G 15:G 16:G 17:G 18:C 19:T 20:A
    tokenized = tokenizer.encode(seq)
    assert tokenized.rle_run_lengths[6:11] == [5, 5, 5, 5, 5]
    assert tokenized.rle_run_lengths[13:18] == [5, 5, 5, 5, 5]
    assert tokenized.rle_base_ids[6] == RLE_BASE_TO_IDX["A"]
    assert tokenized.rle_base_ids[13] == RLE_BASE_TO_IDX["G"]


def test_internal_ambiguous_base_produces_unk_but_tracks_rle(tokenizer):
    seq = "ACGTACNACGTAC"
    tokenized = tokenizer.encode(seq)
    n_index = seq.index("N")
    assert tokenized.token_ids[n_index] == UNK_IDX
    assert tokenized.rle_base_ids[n_index] == RLE_BASE_TO_IDX["N"]
    assert tokenizer.decode(tokenized.token_ids)[n_index] == "N"


def test_rle_uses_unpadded_sequence_not_padded(tokenizer):
    """
    RLE must be computed on the raw, unpadded sequence -- otherwise
    boundary padding (even edge-replicated padding) would silently inflate
    a genuine leading/trailing homopolymer run's reported length.
    """
    seq = "AACGTACGTAC"  # genuine leading run of 2 A's
    tokenized = tokenizer.encode(seq)
    assert tokenized.rle_run_lengths[0] == 2

    padded = tokenizer._pad_sequence(seq)  # "AA" + seq -> leading run becomes 4 if computed here
    wrong_base_ids, wrong_run_lengths = tokenizer._compute_rle(padded)
    assert wrong_run_lengths[LEFT_PAD_BASES] == 4, (
        "This assertion documents the bug that would occur if RLE were computed on the "
        "padded string -- it is expected to differ from the correct value above."
    )


def test_sos_eos_wrap_and_strip(tokenizer):
    seq = "ACGTACGTAC"
    tokenized = tokenizer.encode(seq)
    wrapped = tokenizer.add_special_tokens(tokenized.token_ids)
    assert tokenizer.decode(wrapped) == seq
    # decode must stop at EOS, ignoring anything appended after (e.g. batch padding)
    from config import PAD_IDX

    assert tokenizer.decode(wrapped + [PAD_IDX, PAD_IDX]) == seq


def test_rejects_invalid_alphabet(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.encode("ACGTXYZ")


def test_rejects_empty_sequence(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.encode("")
