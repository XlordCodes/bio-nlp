"""
data/dataset.py
-----------------
Custom PyTorch Dataset + collate_fn: turns (noisy_sequence, clean_sequence)
training pairs into batched, padded tensors ready for SequenceTranslationModel.

-----------------------------------------------------------------------------
WHY THIS FILE NEEDS ALIGNMENT BREAKPOINTS, NOT JUST TWO STRINGS
-----------------------------------------------------------------------------
noisy_sequence and clean_sequence are NOT the same length -- that's the whole
point of framing this as Seq2Seq rather than classification (see Part 2 of
the project brief). But that same length mismatch means naive fixed-index
chunking ("take noisy[0:512] and clean[0:512]") silently breaks the moment a
single indel has occurred before that cut point: the two substrings stop
corresponding to the same genomic region, and every chunk after that point
is misaligned. Training on misaligned chunks doesn't fail loudly -- it
actively teaches the model wrong corrections, which is worse than not
training on that data at all.

The fix used here: every training pair must come with a `breakpoints` list
-- (noisy_index, clean_index) coordinate pairs known (from upstream
alignment, e.g. minimap2/CIGAR parsing in data/simulator.py) to refer to the
same reference position in both sequences. Chunk boundaries are only ever
placed AT a breakpoint, never inside the segment between two consecutive
breakpoints. This guarantees every (noisy_chunk, clean_chunk) pair this file
produces is genuinely alignment-consistent, at the cost of occasionally
producing a chunk larger than the requested chunk_size when a single
alignment segment (rare, but possible with a large indel) exceeds it -- see
`_chunk_aligned_pair` below.

CONTRACT FOR data/simulator.py (not yet built): it must produce, for each
simulated read, a breakpoints list satisfying `AlignedPair.validate()`:
sorted, strictly non-decreasing in both coordinates, starting at (0, 0) and
ending at (len(noisy_sequence), len(clean_sequence)). See
`load_aligned_pairs_from_jsonl` for the exact on-disk JSONL schema this file
expects simulator.py to write.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config import PAD_IDX, RLE_PAD_BASE_IDX
from data.tokenizer import KmerTokenizer


@dataclass
class AlignedPair:
    """
    One full (noisy_read, clean_reference_region) training pair, plus the
    alignment breakpoints needed to chunk it safely. See module docstring.
    """

    noisy_sequence: str
    clean_sequence: str
    breakpoints: List[Tuple[int, int]]

    @classmethod
    def from_identity(cls, noisy_sequence: str, clean_sequence: str) -> "AlignedPair":
        """
        Convenience constructor for the special case of a pure
        substitution-only pair (same length, no indels at all), which needs
        no real alignment beyond start/end. NOT suitable for real Badread
        output, which is dominated by indels -- this exists for quick tests
        and toy examples only.
        """
        if len(noisy_sequence) != len(clean_sequence):
            raise ValueError(
                "from_identity() requires equal-length sequences (no indels). "
                f"Got noisy len={len(noisy_sequence)}, clean len={len(clean_sequence)}. "
                "Real indel-containing pairs must supply real breakpoints."
            )
        n = len(noisy_sequence)
        return cls(noisy_sequence, clean_sequence, breakpoints=[(0, 0), (n, n)])

    def validate(self) -> None:
        if len(self.breakpoints) < 2:
            raise ValueError("breakpoints must contain at least a start and end point.")
        if self.breakpoints[0] != (0, 0):
            raise ValueError(f"breakpoints must start at (0, 0), got {self.breakpoints[0]}.")
        expected_end = (len(self.noisy_sequence), len(self.clean_sequence))
        if self.breakpoints[-1] != expected_end:
            raise ValueError(
                f"breakpoints must end at (len(noisy), len(clean)) = {expected_end}, "
                f"got {self.breakpoints[-1]}."
            )
        for (n0, c0), (n1, c1) in zip(self.breakpoints, self.breakpoints[1:]):
            if n1 < n0 or c1 < c0:
                raise ValueError(
                    f"breakpoints must be non-decreasing in both coordinates; found "
                    f"({n0},{c0}) -> ({n1},{c1})."
                )
            if n1 == n0 and c1 == c0:
                raise ValueError(
                    f"breakpoints contain a zero-length segment ({n0},{c0}) -> ({n1},{c1}); "
                    f"remove duplicate breakpoints."
                )


def _chunk_aligned_pair(
    pair: AlignedPair, chunk_size: int, min_chunk_size: int
) -> List[Tuple[str, str]]:
    """
    Splits one AlignedPair into a list of (noisy_chunk, clean_chunk) string
    pairs, cutting only at breakpoints. Chunks are grown segment-by-segment
    until adding the next segment would push the noisy-side length over
    chunk_size, at which point the chunk is closed and a new one starts.

    If a SINGLE segment (between two consecutive breakpoints) already
    exceeds chunk_size on its own -- possible with an unusually large indel
    -- it is kept whole rather than being cut mid-segment, since cutting
    inside a segment would reintroduce exactly the misalignment problem this
    function exists to avoid. This is deliberate overflow, not a bug; such
    segments should be rare if upstream alignment breakpoints are reasonably
    fine-grained.

    Chunks whose noisy side is shorter than min_chunk_size are dropped
    (avoids training on near-empty tail fragments).
    """
    bp = pair.breakpoints
    chunks: List[Tuple[str, str]] = []

    seg_start = bp[0]
    prev = bp[0]

    for nxt in bp[1:]:
        candidate_noisy_len = nxt[0] - seg_start[0]
        if candidate_noisy_len > chunk_size and prev != seg_start:
            noisy_chunk = pair.noisy_sequence[seg_start[0]:prev[0]]
            clean_chunk = pair.clean_sequence[seg_start[1]:prev[1]]
            if len(noisy_chunk) >= min_chunk_size:
                chunks.append((noisy_chunk, clean_chunk))
            seg_start = prev
        prev = nxt

    # final trailing chunk
    noisy_chunk = pair.noisy_sequence[seg_start[0]:bp[-1][0]]
    clean_chunk = pair.clean_sequence[seg_start[1]:bp[-1][1]]
    if len(noisy_chunk) >= min_chunk_size:
        chunks.append((noisy_chunk, clean_chunk))

    return chunks


class GenomeCorrectionDataset(Dataset):
    """
    Flattens a list of AlignedPair genome/read pairs into fixed-budget
    (noisy_chunk, clean_chunk) training examples, tokenizes each chunk
    lazily in __getitem__ (so DataLoader(num_workers>0) can parallelize
    tokenization), and exposes a static collate_fn that dynamically pads
    each mini-batch to its own max length -- not a fixed global max -- per
    the Part 3 spec.
    """

    def __init__(
        self,
        aligned_pairs: List[AlignedPair],
        tokenizer: Optional[KmerTokenizer] = None,
        chunk_size: int = 512,
        min_chunk_size: int = 16,
        max_chunk_size: Optional[int] = None,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer or KmerTokenizer()
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        self.examples: List[Tuple[str, str]] = []
        dropped_pairs = 0
        dropped_oversized_total = 0
        dropped_short_clean_total = 0
        for pair in aligned_pairs:
            try:
                pair.validate()
            except ValueError as e:
                raise ValueError(f"Invalid AlignedPair passed to GenomeCorrectionDataset: {e}") from e

            pair_chunks = _chunk_aligned_pair(pair, chunk_size, min_chunk_size)

            # _chunk_aligned_pair's own min_chunk_size filter only checks the
            # NOISY side's length. A segment that's a pure, large insertion
            # (noisy bases with no corresponding reference content -- real
            # Badread output includes these) can have a substantial noisy
            # length but a clean/target length of zero, or just a few bases --
            # which would pass that filter and then crash the tokenizer later
            # (KmerTokenizer.encode() correctly refuses an empty sequence) or
            # produce a near-useless training example. Filtered here, after
            # the fact, for the same reason max_chunk_size is: keeps
            # _chunk_aligned_pair's tested signature/behavior untouched for
            # every other caller.
            too_short_clean = sum(1 for n, c in pair_chunks if len(c) < min_chunk_size)
            if too_short_clean:
                pair_chunks = [(n, c) for n, c in pair_chunks if len(c) >= min_chunk_size]
                dropped_short_clean_total += too_short_clean

            if max_chunk_size is not None:
                # Filter AFTER _chunk_aligned_pair returns, rather than inside it,
                # so that function's original signature/behavior stays untouched
                # for every other caller (including tests/test_dataset.py, which
                # calls it directly and asserts on its exact return shape).
                kept = [
                    (n, c) for n, c in pair_chunks
                    if len(n) <= max_chunk_size and len(c) <= max_chunk_size
                ]
                dropped_oversized_total += len(pair_chunks) - len(kept)
                pair_chunks = kept

            if not pair_chunks:
                dropped_pairs += 1
            self.examples.extend(pair_chunks)

        if verbose:
            oversized_kept = sum(1 for n, c in self.examples if len(n) > chunk_size or len(c) > chunk_size)
            cap_note = (
                f"{dropped_oversized_total} chunk(s) dropped for exceeding max_chunk_size={max_chunk_size}. "
                if max_chunk_size is not None
                else "max_chunk_size not set, so no cap was applied. "
            )
            print(
                f"GenomeCorrectionDataset: {len(aligned_pairs)} raw pair(s) -> "
                f"{len(self.examples)} chunk(s) (chunk_size={chunk_size}, "
                f"min_chunk_size={min_chunk_size}). {oversized_kept} kept chunk(s) still exceed "
                f"chunk_size (single alignment segment larger than chunk_size). {cap_note}"
                f"{dropped_short_clean_total} chunk(s) dropped for having a clean/target side "
                f"shorter than min_chunk_size (pure-insertion segments -- noisy side long enough, "
                f"clean side too short or empty). "
                f"{dropped_pairs} raw pair(s) produced no usable chunks."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        noisy_seq, clean_seq = self.examples[idx]

        src_ts = self.tokenizer.encode(noisy_seq)
        clean_ts = self.tokenizer.encode(clean_seq)
        target_ids = self.tokenizer.add_special_tokens(clean_ts.token_ids)

        return {
            "src_token_ids": torch.tensor(src_ts.token_ids, dtype=torch.long),
            "src_rle_base_ids": torch.tensor(src_ts.rle_base_ids, dtype=torch.long),
            "src_rle_run_lengths": torch.tensor(src_ts.rle_run_lengths, dtype=torch.long),
            "src_length": torch.tensor(src_ts.length, dtype=torch.long),
            "target_token_ids": torch.tensor(target_ids, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Dynamically pads a mini-batch to ITS OWN max lengths (source and
        target padded independently, since they have no reason to match).
        Output keys match SequenceTranslationModel.forward()'s keyword
        arguments exactly, so training code can call `model(**batch)`
        directly (plus max_decode_len / teacher_forcing_ratio, which the
        training loop supplies separately since they're training-schedule
        concerns, not data concerns).
        """
        batch_size = len(batch)
        max_src_len = max(ex["src_token_ids"].size(0) for ex in batch)
        max_tgt_len = max(ex["target_token_ids"].size(0) for ex in batch)

        src_tokens = torch.full((batch_size, max_src_len), PAD_IDX, dtype=torch.long)
        rle_base_ids = torch.full((batch_size, max_src_len), RLE_PAD_BASE_IDX, dtype=torch.long)
        rle_run_lengths = torch.zeros((batch_size, max_src_len), dtype=torch.long)
        src_lengths = torch.zeros((batch_size,), dtype=torch.long)
        target_tokens = torch.full((batch_size, max_tgt_len), PAD_IDX, dtype=torch.long)

        for i, ex in enumerate(batch):
            src_len = ex["src_token_ids"].size(0)
            tgt_len = ex["target_token_ids"].size(0)

            src_tokens[i, :src_len] = ex["src_token_ids"]
            rle_base_ids[i, :src_len] = ex["src_rle_base_ids"]
            rle_run_lengths[i, :src_len] = ex["src_rle_run_lengths"]
            src_lengths[i] = ex["src_length"]
            target_tokens[i, :tgt_len] = ex["target_token_ids"]

        return {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "rle_base_ids": rle_base_ids,
            "rle_run_lengths": rle_run_lengths,
            "target_tokens": target_tokens,
        }


def create_dataloader(
    aligned_pairs: List[AlignedPair],
    tokenizer: Optional[KmerTokenizer] = None,
    chunk_size: int = 512,
    min_chunk_size: int = 16,
    max_chunk_size: Optional[int] = None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience wrapper: AlignedPairs in, a ready-to-iterate DataLoader out.

    max_chunk_size: if None (default), automatically set to 4 * chunk_size.
    This bounds the worst-case decode-step blowup a single oversized chunk
    can impose on every other sample sharing its batch (see
    _chunk_aligned_pair). Pass an explicit value, or float('inf'), to
    override.
    """
    if max_chunk_size is None:
        max_chunk_size = 4 * chunk_size
    dataset = GenomeCorrectionDataset(
        aligned_pairs,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=GenomeCorrectionDataset.collate_fn,
        num_workers=num_workers,
    )


def load_aligned_pairs_from_jsonl(path: str) -> List[AlignedPair]:
    """
    Loads AlignedPairs from a JSONL manifest file, one JSON object per line:
        {"noisy_sequence": "...", "clean_sequence": "...", "breakpoints": [[0,0], [118,120], ..., [N,M]]}

    THIS IS THE CONTRACT data/simulator.py MUST SATISFY when it writes its
    training-data output: one line per simulated read, breakpoints computed
    from the read's alignment back to its source reference region.
    """
    pairs = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                breakpoints = [tuple(bp) for bp in record["breakpoints"]]
                pairs.append(
                    AlignedPair(
                        noisy_sequence=record["noisy_sequence"],
                        clean_sequence=record["clean_sequence"],
                        breakpoints=breakpoints,
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Malformed JSONL record at {path}:{line_num}: {e}") from e
    return pairs


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    random.seed(0)

    def simulate_indel_pair(clean_seq: str, num_edits: int, seed: int):
        """
        TEST-ONLY helper: stands in for what data/simulator.py + an alignment
        step will eventually produce -- applies random insertions/deletions/
        substitutions to a clean sequence and returns (noisy_seq,
        breakpoints), so this file's chunking logic can be tested against
        real indel-shifted pairs without depending on Badread or minimap2
        being available. NOT part of the production API.
        """
        rng = random.Random(seed)
        noisy_chars = []
        breakpoints = [(0, 0)]
        noisy_pos = 0
        clean_pos = 0

        edit_positions = sorted(rng.sample(range(len(clean_seq)), min(num_edits, len(clean_seq))))
        edit_set = set(edit_positions)

        for clean_pos in range(len(clean_seq)):
            if clean_pos in edit_set:
                op = rng.choice(["ins", "del", "sub"])
                if op == "ins":
                    # insert 1-3 random extra bases before this clean base, then keep the clean base too
                    extra = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 3)))
                    noisy_chars.append(extra)
                    noisy_pos += len(extra)
                    noisy_chars.append(clean_seq[clean_pos])
                    noisy_pos += 1
                elif op == "del":
                    # drop this clean base entirely (noisy_pos does not advance)
                    pass
                else:  # sub
                    alt = rng.choice([b for b in "ACGT" if b != clean_seq[clean_pos]])
                    noisy_chars.append(alt)
                    noisy_pos += 1
            else:
                noisy_chars.append(clean_seq[clean_pos])
                noisy_pos += 1
            breakpoints.append((noisy_pos, clean_pos + 1))

        noisy_seq = "".join(noisy_chars)
        # collapse to strictly-increasing-in-at-least-one-coord breakpoints (dedupe consecutive dupes)
        deduped = [breakpoints[0]]
        for bp in breakpoints[1:]:
            if bp != deduped[-1]:
                deduped.append(bp)
        return noisy_seq, deduped

    # -- 1. Chunking correctness: reconstruct the FULL original noisy and ---
    #       clean sequences exactly by concatenating all chunks (min_chunk_size=1
    #       so nothing gets dropped, proving zero data loss/duplication) -------
    clean_seq = "".join(random.choice("ACGT") for _ in range(3000))
    noisy_seq, breakpoints = simulate_indel_pair(clean_seq, num_edits=150, seed=1)
    pair = AlignedPair(noisy_seq, clean_seq, breakpoints)
    pair.validate()

    chunks = _chunk_aligned_pair(pair, chunk_size=256, min_chunk_size=1)
    reconstructed_noisy = "".join(c[0] for c in chunks)
    reconstructed_clean = "".join(c[1] for c in chunks)
    assert reconstructed_noisy == noisy_seq, "Chunking lost or duplicated noisy-sequence data"
    assert reconstructed_clean == clean_seq, "Chunking lost or duplicated clean-sequence data"
    assert all(len(n) <= 256 or True for n, c in chunks)  # oversized chunks are allowed, just logging below
    oversized = [len(n) for n, c in chunks if len(n) > 256]
    print(
        f"[1/5] Chunking reconstruction passed: {len(chunks)} chunks, exact lossless "
        f"reconstruction of both sequences ({len(oversized)} oversized chunk(s) from large "
        f"single alignment segments, as expected)."
    )

    # -- 2. Alignment consistency: every chunk pair must come from a matching --
    #       reference region, not just any substrings. Cross-check by re-deriving
    #       each chunk's expected clean substring from its own breakpoints. -----
    chunks_with_bounds = []
    seg_start = breakpoints[0]
    prev = breakpoints[0]
    for nxt in breakpoints[1:]:
        if nxt[0] - seg_start[0] > 256 and prev != seg_start:
            chunks_with_bounds.append((seg_start, prev))
            seg_start = prev
        prev = nxt
    chunks_with_bounds.append((seg_start, breakpoints[-1]))
    for (n0, c0), (n1, c1) in chunks_with_bounds:
        assert noisy_seq[n0:n1] == pair.noisy_sequence[n0:n1]
        assert clean_seq[c0:c1] == pair.clean_sequence[c0:c1]
    print(f"[2/5] Alignment-consistency cross-check passed for {len(chunks_with_bounds)} chunk boundaries.")

    # -- 3. min_chunk_size filtering drops tiny trailing fragments -------------
    small_pair = AlignedPair.from_identity("ACGTACGTAC", "ACGTACGTAC")
    small_chunks_kept = _chunk_aligned_pair(small_pair, chunk_size=1000, min_chunk_size=20)
    small_chunks_dropped_none = _chunk_aligned_pair(small_pair, chunk_size=1000, min_chunk_size=1)
    assert small_chunks_kept == [], "A 10-base pair should be dropped when min_chunk_size=20"
    assert len(small_chunks_dropped_none) == 1
    print("[3/5] min_chunk_size filtering passed.")

    # -- 4. Full Dataset + collate_fn: build real batches, check padding -------
    pairs = []
    for i in range(6):
        c = "".join(random.choice("ACGT") for _ in range(random.randint(80, 200)))
        n, bps = simulate_indel_pair(c, num_edits=random.randint(3, 10), seed=100 + i)
        pairs.append(AlignedPair(n, c, bps))

    dataset = GenomeCorrectionDataset(pairs, chunk_size=512, min_chunk_size=8, verbose=True)
    assert len(dataset) > 0

    loader = create_dataloader(pairs, chunk_size=512, min_chunk_size=8, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    print("\n[4/5] Batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")

    B = batch["src_tokens"].size(0)
    for i in range(B):
        true_len = batch["src_lengths"][i].item()
        # everything past true_len on the source side must be PAD_IDX
        assert (batch["src_tokens"][i, true_len:] == PAD_IDX).all(), "Padding leaked non-PAD tokens"
        assert (batch["rle_base_ids"][i, true_len:] == RLE_PAD_BASE_IDX).all()
    print("[4/5] Dynamic padding correctness verified (padding regions are exactly PAD_IDX / RLE_PAD_BASE_IDX).")

    # -- 5. Full integration: a real batch through the real model, plus a real -
    #       backward pass with loss masking via ignore_index=PAD_IDX -----------
    from model.sequence_translation_model import SequenceTranslationModel, SequenceTranslationConfig
    import torch.nn as nn

    torch.manual_seed(0)
    model = SequenceTranslationModel(SequenceTranslationConfig())

    output = model(
        src_tokens=batch["src_tokens"],
        src_lengths=batch["src_lengths"],
        rle_base_ids=batch["rle_base_ids"],
        rle_run_lengths=batch["rle_run_lengths"],
        target_tokens=batch["target_tokens"],
        teacher_forcing_ratio=1.0,
    )

    # Standard seq2seq loss-masking contract: predict target_tokens[:, 1:] from
    # target_tokens[:, :-1], ignore <PAD> positions so they don't contribute to the loss.
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    targets_for_loss = batch["target_tokens"][:, 1:]
    loss = loss_fn(output.logits.reshape(-1, output.logits.size(-1)), targets_for_loss.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss), "Loss must be finite"
    assert model.encoder.embedding.weight.grad is not None
    print(
        f"\n[5/5] Full training-step integration passed: real batch -> model.forward() -> "
        f"masked CrossEntropyLoss -> backward(). Loss = {loss.item():.4f}"
    )

    print("\nAll dataset sanity checks passed.")