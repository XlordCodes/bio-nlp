"""
data/tokenizer.py
-------------------
Bio-NLP preprocessing: raw nucleotide strings <-> hexamer token ids, plus the
per-position RLE (Run-Length-Encoded homopolymer) auxiliary channel the
decoder needs.

Scope boundary: this file assumes it is handed a plain nucleotide string
(already stripped of FASTA headers / multi-record structure). Parsing actual
.fasta/.fa files with headers lives in data/simulator.py (for training data)
and backend/inference_engine.py (for inference requests) -- neither of those
should reimplement k-mer or RLE logic; they should call into this file.

-----------------------------------------------------------------------------
THE CENTER-BASE MAPPING / PADDING DERIVATION (read this before touching K)
-----------------------------------------------------------------------------
We need every hexamer token to have exactly one "anchor" base, and we need
exactly N tokens for an N-base sequence (one token per original base -- this
is what lets rle_base_ids / rle_run_lengths line up 1:1 with token_ids by
plain index, with no separate coordinate-mapping step anywhere downstream).

The anchor is defined as the base at 0-indexed offset 2 within each 6-base
window (config.RLE_ANCHOR_OFFSET) -- the 3rd base of the window, counting
from 1.

For hexamer window `start` (0-indexed, spanning padded[start : start+6]) to
have its anchor land exactly on padded[start + 2], and for that in turn to
equal the true original base at index `start` when start=0, we need exactly
2 padding bases prepended before the sequence starts (LEFT_PAD_BASES = 2).

Since k=6 is even, k-1=5 is odd, and an odd amount of total padding cannot be
split evenly between both ends. The split is therefore deliberately
asymmetric: LEFT_PAD_BASES=2, RIGHT_PAD_BASES=3 (config.py). This is the
only split that (a) keeps the left anchor alignment correct AND (b) produces
exactly N windows for an N-base sequence.

Padding is applied by EDGE-REPLICATING the sequence's own boundary base
(repeating sequence[0] on the left, sequence[-1] on the right), NOT by
inserting a fixed 'N' placeholder. This distinction matters and was caught
by this file's own sanity checks during development: padding with literal
'N' makes every window that touches the padding -- the first
LEFT_PAD_BASES and last RIGHT_PAD_BASES tokens of *every single read or
chunk* -- unrepresentable in the pure-ACGT k-mer vocabulary, forcing UNK
there regardless of whether the sequence is actually ambiguous at that
position, and making that information unrecoverable on decode. Edge
replication keeps the exact same padding COUNT (so the anchor-offset
derivation above is unchanged) while keeping boundary windows valid,
embeddable k-mers whenever the true boundary base is unambiguous.

Both properties are proven and checked at runtime in `encode()` and
exercised in this file's `__main__` sanity checks below -- including a
check that demonstrates why RLE must be computed on the *unpadded*
sequence rather than the padded one (padding characters, even
edge-replicated ones, would otherwise silently inflate a genuine
leading/trailing homopolymer run).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from config import (
    K,
    NUM_KMERS,
    VOCAB_SIZE,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    UNK_IDX,
    NUM_SPECIAL_TOKENS,
    NUCLEOTIDES,
    VALID_INPUT_CHARS,
    RLE_BASE_TO_IDX,
    RLE_ANCHOR_OFFSET,
    LEFT_PAD_BASES,
    RIGHT_PAD_BASES,
)

# Fail fast at import time if config.py's padding geometry has been edited
# inconsistently -- every method below silently relies on this being true.
assert LEFT_PAD_BASES + RIGHT_PAD_BASES == K - 1, (
    "config.py: LEFT_PAD_BASES + RIGHT_PAD_BASES must equal K - 1, otherwise "
    "encode() will not produce exactly one token per original base."
)
assert RLE_ANCHOR_OFFSET == LEFT_PAD_BASES, (
    "config.py: RLE_ANCHOR_OFFSET must equal LEFT_PAD_BASES, otherwise window "
    "0's anchor will not land on original base 0."
)

NUCLEOTIDE_TO_DIGIT = {base: digit for digit, base in enumerate(NUCLEOTIDES)}  # A=0, C=1, G=2, T=3


@dataclass
class TokenizedSequence:
    """Result of tokenizing a single raw nucleotide string."""
    token_ids: List[int]        # length N; hexamer token ids (offset by NUM_SPECIAL_TOKENS), UNK where ambiguous
    rle_base_ids: List[int]     # length N; aligned 1:1 with token_ids by index
    rle_run_lengths: List[int]  # length N; aligned 1:1 with token_ids by index
    length: int                 # == N == len(original, unpadded sequence)


class KmerTokenizer:
    """
    Stateless (no learned/fitted parameters) hexamer tokenizer. k-mer id <->
    string conversion is done arithmetically (base-4 positional encoding over
    A/C/G/T), not via a stored lookup table, since it's a pure bijection.
    """

    def __init__(self, k: int = K):
        if k != K:
            raise ValueError(
                f"This tokenizer's padding/anchor derivation (LEFT_PAD_BASES, "
                f"RIGHT_PAD_BASES, RLE_ANCHOR_OFFSET in config.py) is derived "
                f"specifically for k={K}. Changing k to {k} requires re-deriving "
                f"those constants first -- see the module docstring above."
            )
        self.k = k

    # -- validation ---------------------------------------------------------

    def _validate_sequence(self, sequence: str) -> None:
        if len(sequence) == 0:
            raise ValueError("Cannot tokenize an empty sequence.")
        invalid_chars = set(sequence) - VALID_INPUT_CHARS
        if invalid_chars:
            raise ValueError(
                f"Sequence contains invalid character(s) {sorted(invalid_chars)}; "
                f"only {sorted(VALID_INPUT_CHARS)} are permitted. (Call "
                f"sequence.upper() before encode() if your input is lowercase.)"
            )

    # -- k-mer <-> id arithmetic ---------------------------------------------

    def _kmer_str_to_id(self, kmer: str) -> Optional[int]:
        """
        Pure-ACGT hexamer -> integer id in [0, NUM_KMERS). Returns None if the
        window contains any non-ACGT character (an 'N' from either a
        low-confidence basecall or boundary padding) -- callers map None to
        UNK_IDX, since such a window has no representation in the 4-symbol
        k-mer vocabulary by construction (|V| = 4^K assumes strictly A/C/G/T).
        """
        kmer_id = 0
        for ch in kmer:
            digit = NUCLEOTIDE_TO_DIGIT.get(ch)
            if digit is None:
                return None
            kmer_id = kmer_id * 4 + digit
        return kmer_id

    def _kmer_id_to_str(self, kmer_id: int) -> str:
        if not (0 <= kmer_id < NUM_KMERS):
            raise ValueError(f"kmer_id {kmer_id} out of range [0, {NUM_KMERS}).")
        digits = []
        remaining = kmer_id
        for _ in range(self.k):
            digits.append(NUCLEOTIDES[remaining % 4])
            remaining //= 4
        return "".join(reversed(digits))

    # -- padding --------------------------------------------------------------

    def _pad_sequence(self, sequence: str) -> str:
        """
        Edge-replicate padding: repeats the sequence's own first base
        LEFT_PAD_BASES times on the 5' end, and its own last base
        RIGHT_PAD_BASES times on the 3' end. See module docstring for why
        this is used instead of a fixed 'N' placeholder.
        """
        left_pad = sequence[0] * LEFT_PAD_BASES
        right_pad = sequence[-1] * RIGHT_PAD_BASES
        return left_pad + sequence + right_pad

    # -- RLE ------------------------------------------------------------------

    def _compute_rle(self, sequence: str) -> Tuple[List[int], List[int]]:
        """
        Run-length encodes the ORIGINAL, UNPADDED sequence -- deliberately NOT
        the padded one. If we ran this over the padded string, a real leading
        or trailing homopolymer run (e.g. a genuine run of N's from low
        basecall confidence right at the start of a read) would silently
        absorb the synthetic boundary-padding characters into the same run,
        inflating its reported length. Computing RLE on the unpadded sequence
        and then relying on the anchor guarantee (token i's anchor == orig[i])
        avoids this entirely; see the __main__ block for a concrete before/after
        demonstration.
        """
        n = len(sequence)
        base_ids = [0] * n
        run_lengths = [0] * n
        i = 0
        while i < n:
            j = i
            while j < n and sequence[j] == sequence[i]:
                j += 1
            run_len = j - i
            base_id = RLE_BASE_TO_IDX[sequence[i]]  # sequence already validated against VALID_INPUT_CHARS
            for p in range(i, j):
                base_ids[p] = base_id
                run_lengths[p] = run_len
            i = j
        return base_ids, run_lengths

    # -- public API -------------------------------------------------------------

    def encode(self, sequence: str) -> TokenizedSequence:
        """
        Raw nucleotide string -> TokenizedSequence. Uppercases the input, then
        validates it against {A, C, G, T, N}.
        """
        sequence = sequence.upper()
        self._validate_sequence(sequence)
        n = len(sequence)

        padded = self._pad_sequence(sequence)
        num_windows = len(padded) - self.k + 1
        assert num_windows == n, (
            f"Padding geometry produced {num_windows} hexamer windows for a "
            f"{n}-base sequence; expected exactly {n}. This should be "
            f"mathematically impossible given the config.py invariants checked "
            f"at import time -- if you see this, something is patching those "
            f"constants at runtime."
        )

        token_ids: List[int] = []
        for start in range(n):
            window = padded[start:start + self.k]
            kmer_id = self._kmer_str_to_id(window)
            if kmer_id is None:
                token_ids.append(UNK_IDX)  # window touches an 'N' -- ambiguous base or boundary padding
            else:
                token_ids.append(kmer_id + NUM_SPECIAL_TOKENS)

        rle_base_ids, rle_run_lengths = self._compute_rle(sequence)

        return TokenizedSequence(
            token_ids=token_ids,
            rle_base_ids=rle_base_ids,
            rle_run_lengths=rle_run_lengths,
            length=n,
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        """
        Reconstructs a nucleotide string from a sequence of hexamer token ids
        by reading off each token's ANCHOR base (offset RLE_ANCHOR_OFFSET
        within its decoded 6-mer string).

        This works, and requires no overlap-consensus/stitching logic between
        adjacent predicted tokens, specifically BECAUSE encode() guarantees
        token i's anchor equals original base i. A decoder trained to predict
        tokens under this same tokenization scheme can therefore be decoded
        position-by-position -- we do not need consecutive predicted hexamers
        to agree with each other on their overlapping regions (they may not,
        for a partially-trained or imperfect model) because only each token's
        own anchor character is ever read.

        Special-token handling: <PAD>/<SOS> are skipped, <EOS> stops decoding,
        <UNK> emits 'N' (an unrecoverable/ambiguous base, consistent with
        standard bioinformatics convention).
        """
        bases: List[str] = []
        for tid in token_ids:
            if tid == PAD_IDX or tid == SOS_IDX:
                continue
            if tid == EOS_IDX:
                break
            if tid == UNK_IDX:
                bases.append("N")
                continue
            kmer_id = tid - NUM_SPECIAL_TOKENS
            if not (0 <= kmer_id < NUM_KMERS):
                raise ValueError(
                    f"Token id {tid} is outside the valid vocabulary range [0, {VOCAB_SIZE})."
                )
            kmer_str = self._kmer_id_to_str(kmer_id)
            bases.append(kmer_str[RLE_ANCHOR_OFFSET])
        return "".join(bases)

    def decode_with_confidence(
        self, token_ids: Sequence[int], step_confidences: Sequence[float]
    ) -> Tuple[str, List[float]]:
        """
        Like decode(), but also returns a per-output-base confidence value --
        typically the softmax probability the model assigned to the token it
        actually chose at that decode step. Mirrors decode()'s exact
        skip/break control flow (PAD/SOS skipped, EOS stops, UNK emits 'N')
        so confidences[i] always corresponds to the same output base as
        bases[i]; kept in this one place rather than duplicated at the call
        site so the two can never silently drift apart.

        step_confidences must be the same length as token_ids (one value per
        decode step, aligned index-for-index -- typically max softmax
        probability at that step).
        """
        if len(step_confidences) != len(token_ids):
            raise ValueError(
                f"step_confidences length ({len(step_confidences)}) must match "
                f"token_ids length ({len(token_ids)})."
            )
        bases: List[str] = []
        confidences: List[float] = []
        for tid, conf in zip(token_ids, step_confidences):
            if tid == PAD_IDX or tid == SOS_IDX:
                continue
            if tid == EOS_IDX:
                break
            if tid == UNK_IDX:
                bases.append("N")
                confidences.append(conf)
                continue
            kmer_id = tid - NUM_SPECIAL_TOKENS
            if not (0 <= kmer_id < NUM_KMERS):
                raise ValueError(
                    f"Token id {tid} is outside the valid vocabulary range [0, {VOCAB_SIZE})."
                )
            kmer_str = self._kmer_id_to_str(kmer_id)
            bases.append(kmer_str[RLE_ANCHOR_OFFSET])
            confidences.append(conf)
        return "".join(bases), confidences

    def add_special_tokens(self, token_ids: Sequence[int]) -> List[int]:
        """
        Wraps a raw token id sequence with a leading <SOS> and trailing <EOS>.
        This is the ONLY place SOS/EOS should be added -- e.g. when preparing
        decoder target_tokens for training. Callers should not prepend/append
        them manually elsewhere, to keep exactly one source of truth for the
        wrapping convention.
        """
        return [SOS_IDX] + list(token_ids) + [EOS_IDX]

    def encode_to_tensors(self, sequence: str) -> dict:
        """
        Convenience wrapper: raw string -> ready-to-batch torch tensors
        (unbatched, i.e. shape (L,) not (1, L) -- callers add the batch dim).
        This is the "Tensor Conversion" step in the Part 4 inference pipeline
        diagram: [Raw String] -> [Tokenizer] -> [Tensor Conversion].
        """
        ts = self.encode(sequence)
        return {
            "token_ids": torch.tensor(ts.token_ids, dtype=torch.long),
            "rle_base_ids": torch.tensor(ts.rle_base_ids, dtype=torch.long),
            "rle_run_lengths": torch.tensor(ts.rle_run_lengths, dtype=torch.long),
            "length": torch.tensor(ts.length, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    tokenizer = KmerTokenizer()

    # -- 1. Round-trip on random pure-ACGT sequences of various lengths ------
    random.seed(0)
    for test_len in [1, 5, 6, 7, 50, 200]:
        seq = "".join(random.choice("ACGT") for _ in range(test_len))
        ts = tokenizer.encode(seq)
        assert len(ts.token_ids) == test_len, f"Expected {test_len} tokens, got {len(ts.token_ids)}"
        assert len(ts.rle_base_ids) == test_len
        assert len(ts.rle_run_lengths) == test_len
        assert UNK_IDX not in ts.token_ids, "Pure-ACGT sequence should never produce UNK tokens"

        reconstructed = tokenizer.decode(ts.token_ids)
        assert reconstructed == seq, (
            f"Round-trip failed for length {test_len}:\n  original:      {seq}\n  reconstructed: {reconstructed}"
        )
    print(f"[1/6] Round-trip encode/decode passed for lengths [1, 5, 6, 7, 50, 200].")

    # -- 2. Homopolymer run-length correctness --------------------------------
    seq = "GATTACAAAAATCGGGGGCTA"
    #  index: 0:G 1:A 2:T 3:T 4:A 5:C 6:A 7:A 8:A 9:A 10:A 11:T 12:C
    #         13:G 14:G 15:G 16:G 17:G 18:C 19:T 20:A
    # positions 6-10 = "AAAAA" (run length 5), positions 13-17 = "GGGGG" (run length 5)
    ts = tokenizer.encode(seq)
    assert ts.rle_run_lengths[6:11] == [5, 5, 5, 5, 5], ts.rle_run_lengths[6:11]
    assert ts.rle_run_lengths[13:18] == [5, 5, 5, 5, 5], ts.rle_run_lengths[13:18]
    assert ts.rle_base_ids[6] == RLE_BASE_TO_IDX["A"]
    assert ts.rle_base_ids[13] == RLE_BASE_TO_IDX["G"]
    # single-base runs elsewhere should report length 1
    assert ts.rle_run_lengths[0] == 1  # 'G' in "GATTACA..." is a run of 1
    print("[2/6] Homopolymer run-length correctness passed.")

    # -- 3. Internal ambiguous 'N' produces UNK tokens in the k-mer stream, ---
    #       but is still tracked as a first-class base in the RLE channel.
    seq = "ACGTACNACGTAC"
    ts = tokenizer.encode(seq)
    n_index = seq.index("N")
    assert ts.rle_base_ids[n_index] == RLE_BASE_TO_IDX["N"]
    assert ts.rle_run_lengths[n_index] == 1  # isolated single N
    assert ts.token_ids[n_index] == UNK_IDX, (
        "The window anchored exactly at the N must be UNK (N isn't in the ACGT k-mer vocab)"
    )
    # decode() should fall back to 'N' for every UNK position
    reconstructed = tokenizer.decode(ts.token_ids)
    assert reconstructed[n_index] == "N"
    print("[3/6] Internal 'N' handling (UNK in k-mer stream, first-class in RLE) passed.")

    # -- 4. Concrete demonstration of why RLE must run on the UNPADDED sequence,
    #       not the padded one (edge-replicated padding would inflate a
    #       genuine boundary run, just as literal-N padding would have) -------
    seq_leading_run = "AACGTACGTAC"  # genuine leading run of 2 A's
    ts = tokenizer.encode(seq_leading_run)
    correct_leading_run = ts.rle_run_lengths[0]
    assert correct_leading_run == 2, f"Expected leading A run length 2, got {correct_leading_run}"

    # Now show what the WRONG approach (RLE on the padded string, read at the
    # anchor offset) would have produced instead, to justify computing RLE on
    # the unpadded sequence: edge-replicate padding repeats seq[0]='A' twice,
    # so the padded string's leading run becomes 4 A's instead of the true 2.
    padded_wrong = tokenizer._pad_sequence(seq_leading_run)  # "AA" + "AACGTACGTAC" = "AAAACGTACGTAC"
    wrong_base_ids, wrong_run_lengths = tokenizer._compute_rle(padded_wrong)
    wrong_leading_run_at_anchor = wrong_run_lengths[LEFT_PAD_BASES]  # what position 0's anchor would report
    assert wrong_leading_run_at_anchor == 4, wrong_leading_run_at_anchor  # 2 padding A's + 2 real A's merged
    assert wrong_leading_run_at_anchor != correct_leading_run, (
        "This assertion is EXPECTED to hold -- it's demonstrating the bug we avoided: "
        f"computing RLE on the padded string would have reported a run of "
        f"{wrong_leading_run_at_anchor} instead of the true {correct_leading_run}, even "
        f"with edge-replicate (not literal-N) padding."
    )
    print(
        f"[4/6] Confirmed boundary-contamination bug avoided: RLE-on-padded would report "
        f"run length {wrong_leading_run_at_anchor}, RLE-on-unpadded correctly reports "
        f"{correct_leading_run}."
    )

    # -- 5. SOS/EOS wrapping + decode stripping --------------------------------
    seq = "ACGTACGTAC"
    ts = tokenizer.encode(seq)
    wrapped = tokenizer.add_special_tokens(ts.token_ids)
    assert wrapped[0] == SOS_IDX and wrapped[-1] == EOS_IDX
    assert len(wrapped) == len(ts.token_ids) + 2
    reconstructed = tokenizer.decode(wrapped)
    assert reconstructed == seq, f"Expected '{seq}', got '{reconstructed}' after SOS/EOS wrap+decode"
    # decode must also stop AT EOS, ignoring anything appended after it (e.g. batch padding)
    padded_after_eos = wrapped + [PAD_IDX, PAD_IDX, PAD_IDX]
    assert tokenizer.decode(padded_after_eos) == seq
    print("[5/6] SOS/EOS wrapping, stripping, and post-EOS pad truncation passed.")

    # -- 6. Full integration: tokenizer output actually runs through the real ---
    #       SequenceTranslationModel built in previous files (batch size 1) --
    from model.sequence_translation_model import SequenceTranslationModel, SequenceTranslationConfig

    torch.manual_seed(0)
    seq = "".join(random.choice("ACGT") for _ in range(40))
    tensors = tokenizer.encode_to_tensors(seq)

    model = SequenceTranslationModel(SequenceTranslationConfig())
    output = model.predict(
        src_tokens=tensors["token_ids"].unsqueeze(0),
        src_lengths=tensors["length"].unsqueeze(0),
        rle_base_ids=tensors["rle_base_ids"].unsqueeze(0),
        rle_run_lengths=tensors["rle_run_lengths"].unsqueeze(0),
        max_decode_len=50,
    )
    decoded_sequence = tokenizer.decode(output.predicted_tokens[0].tolist())
    print(
        f"[6/6] Full integration passed: tokenizer -> SequenceTranslationModel.predict() -> "
        f"decode() ran end-to-end. Predicted attention_matrix shape: "
        f"{tuple(output.attention_matrix.shape)}, decoded output length: {len(decoded_sequence)} "
        f"(untrained weights, so output content is expected to be noise -- this test only "
        f"proves the tensors are shape/dtype-compatible end-to-end, not that predictions are correct)."
    )

    print("\nAll tokenizer sanity checks passed.")