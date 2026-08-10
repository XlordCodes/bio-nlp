"""
data/simulator.py
-------------------
Generates real (noisy, clean) training pairs for the genome error-correction
model: simulates ONT-style noisy long reads from a reference genome using
Badread, aligns each read back to the reference with minimap2, and writes
out a JSONL manifest in the exact schema data/dataset.py's
`load_aligned_pairs_from_jsonl` expects.

-----------------------------------------------------------------------------
WHY BADREAD'S OWN READ HEADERS AREN'T ENOUGH
-----------------------------------------------------------------------------
Badread reports each read's approximate source region in its FASTQ header
(e.g. "reference,+strand,14628-17801"), but that's provenance, not an
alignment -- it doesn't say exactly where each individual insertion,
deletion, or substitution landed. data/dataset.py needs exact breakpoints
(see its module docstring) to chunk long reads without misaligning noisy/
clean substrings. So every simulated read is re-aligned to the reference
with minimap2 (via the `mappy` bindings) to get a real CIGAR string, which
is then walked operation-by-operation to build those breakpoints.

Two real complications this file has to handle, found by testing against
actual Badread output rather than assuming a clean case (see chat for the
smoke test this was developed against):

1. SOFT-CLIPPED / UNALIGNED READ ENDS. Badread deliberately injects some
   adapter sequence, chimeric joins, and pure junk/random reads (see its
   own --junk_reads / --random_reads / --chimeras options). These portions
   of a read have no corresponding reference region at all. Reads are
   trimmed to their aligned span [q_st:q_en) before being used; the
   unaligned ends are discarded, not forced into a training pair.

2. STRAND. A read can align to either strand. When it aligns to the
   reverse strand, the reference region has to be reverse-complemented
   before it corresponds base-for-base to the read as sequenced.

Reads with no confident primary alignment (Badread's junk/random reads are
expected to fall in this category) are skipped entirely, not included as
degenerate all-N training pairs.
"""

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import os
import sys

try:
    import mappy as mp
except ImportError as e:
    raise ImportError(
        "mappy (minimap2 Python bindings) is required by data/simulator.py. "
        "Install with: pip install mappy"
    ) from e

from config import DEFAULT_REFERENCE_FASTA_PATH, DEFAULT_TRAINING_DATA_JSONL_PATH

COMPLEMENT_TABLE = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT_TABLE)[::-1]


@dataclass
class SimulatorConfig:
    reference_fasta: str = DEFAULT_REFERENCE_FASTA_PATH
    output_jsonl: str = DEFAULT_TRAINING_DATA_JSONL_PATH
    quantity: str = "50x"                 # Badread --quantity: coverage multiple (e.g. "50x") or absolute bp ("5000000")
    identity: Optional[str] = None        # Badread --identity "mean,max,stdev"; None = Badread's built-in ONT default
    seed: int = 42
    badread_extra_args: List[str] = field(default_factory=list)  # passthrough flags, e.g. ["--chimeras", "0"]
    min_aligned_length: int = 50          # discard alignments shorter than this -- too short to be useful signal


def load_reference(path: str) -> Tuple[str, str]:
    """
    Loads a SINGLE-record reference FASTA. Returns (header, sequence).
    Raises clearly if the file contains more than one record -- this
    pipeline is designed around one reference chromosome (e.g. E. coli
    K-12 MG1655), not a multi-contig assembly.
    """
    header = None
    seq_parts: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    raise ValueError(
                        f"{path} contains more than one FASTA record; data/simulator.py expects "
                        f"a single-record reference genome (e.g. one E. coli chromosome). Split "
                        f"multi-contig references and run this file once per contig."
                    )
                header = line[1:]
            else:
                seq_parts.append(line.upper())
    if header is None:
        raise ValueError(f"{path} contains no FASTA header / no sequence data.")
    return header, "".join(seq_parts)


def run_badread(
    reference_fasta: str, quantity: str, seed: int, identity: Optional[str], extra_args: List[str]
) -> str:
    """
    Invokes Badread as a subprocess to simulate ONT-style noisy long reads.
    Badread writes simulated reads (FASTQ) to stdout and a progress/config
    log to stderr; this function returns the captured stdout text.
    """
    badread_bin = shutil.which("badread")
    if badread_bin is None:
        python_bin = os.path.join(os.path.dirname(sys.executable), "badread")
        user_bin = os.path.expanduser("~/.local/bin/badread")
        if os.path.isfile(python_bin) and os.access(python_bin, os.X_OK):
            badread_bin = python_bin
        elif os.path.isfile(user_bin) and os.access(user_bin, os.X_OK):
            badread_bin = user_bin

    if badread_bin is None:
        raise RuntimeError(
            "The 'badread' command was not found on PATH. Install with: "
            "pip install git+https://github.com/rrwick/Badread.git"
        )

    cmd = [
        badread_bin, "simulate",
        "--reference", reference_fasta,
        "--quantity", quantity,
        "--seed", str(seed),
    ]
    if identity is not None:
        cmd += ["--identity", identity]
    cmd += extra_args

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Badread failed (exit code {result.returncode}). Command: {' '.join(cmd)}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )
    return result.stdout


def parse_fastq(fastq_text: str) -> List[Tuple[str, str]]:
    """
    Minimal FASTQ parser: returns [(header, sequence), ...]. Quality strings
    are intentionally not returned -- this pipeline learns corrections from
    sequence identity via alignment, not from Badread's simulated per-base
    confidence scores.
    """
    lines = fastq_text.strip().split("\n")
    if not lines or lines == [""]:
        return []
    if len(lines) % 4 != 0:
        raise ValueError(f"Malformed FASTQ: expected a multiple of 4 lines, got {len(lines)}.")
    records = []
    for i in range(0, len(lines), 4):
        header = lines[i]
        if not header.startswith("@"):
            raise ValueError(f"Malformed FASTQ at line {i}: expected '@' header, got: {header[:50]!r}")
        seq = lines[i + 1]
        records.append((header[1:], seq))
    return records


def cigar_to_breakpoints(cigar: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Walks a minimap2/mappy CIGAR -- a list of [length, op] pairs using
    standard SAM op codes (0=M, 1=I, 2=D, 7='=', 8='X') -- and emits one
    breakpoint per operation, in coordinates relative to the start of the
    ALIGNED span (i.e. position 0 corresponds to the first base of the
    already-trimmed read/reference, not the original untrimmed read).

    M/=/X advance both query and reference (matches AND mismatches --
    mismatches are exactly the substitution errors we want the model to
    learn to fix, so they stay inside a normal aligned segment rather than
    becoming a breakpoint of their own).
    I (insertion) advances only the query: these bases exist in the noisy
    read with no clean counterpart.
    D (deletion) advances only the reference: these bases exist in the
    clean sequence with no noisy counterpart.
    """
    q_pos, r_pos = 0, 0
    breakpoints = [(0, 0)]
    for length, op in cigar:
        if op in (0, 7, 8):
            q_pos += length
            r_pos += length
        elif op == 1:
            q_pos += length
        elif op == 2:
            r_pos += length
        else:
            continue  # N, S, H, P -- not expected inside a primary hit's core CIGAR
        breakpoints.append((q_pos, r_pos))
    return breakpoints


def align_read_to_reference(
    aligner: "mp.Aligner", read_seq: str, reference_seq: str, min_aligned_length: int
) -> Optional[Tuple[str, str, List[Tuple[int, int]]]]:
    """
    Aligns one noisy read against the reference and returns
    (noisy_trimmed, clean_trimmed, breakpoints) for the best PRIMARY hit,
    or None if the read has no usable alignment.

    "Best" = longest aligned span among primary hits. Supplementary
    alignments (the other half of a chimeric read) are ignored -- only the
    single best-aligning segment of each read is used.
    """
    best_hit = None
    for hit in aligner.map(read_seq):
        if not hit.is_primary:
            continue
        aligned_len = hit.q_en - hit.q_st
        if best_hit is None or aligned_len > (best_hit.q_en - best_hit.q_st):
            best_hit = hit

    if best_hit is None:
        return None
    if (best_hit.q_en - best_hit.q_st) < min_aligned_length:
        return None

    noisy_trimmed = read_seq[best_hit.q_st:best_hit.q_en]
    clean_trimmed = reference_seq[best_hit.r_st:best_hit.r_en]
    if best_hit.strand == -1:
        # Confirmed empirically (see verify_strand_orientation.py, not guessed):
        # mappy's hit.cigar for a reverse-strand hit describes the alignment of
        # REVCOMP(read[q_st:q_en]) against the FORWARD reference span -- not the
        # read as originally given. So the READ side must be reverse-complemented
        # to match what the CIGAR (and therefore the breakpoints below) actually
        # describes; the reference side stays forward-oriented as-is.
        noisy_trimmed = reverse_complement(noisy_trimmed)

    breakpoints = cigar_to_breakpoints(best_hit.cigar)

    expected_end = (len(noisy_trimmed), len(clean_trimmed))
    if breakpoints[-1] != expected_end:
        # Should be mathematically guaranteed by a well-formed CIGAR. Failing
        # loudly here (rather than silently skipping) turns a would-be silent
        # training-data corruption into an immediate, visible bug report.
        raise RuntimeError(
            f"CIGAR walk produced breakpoints ending at {breakpoints[-1]}, but trimmed "
            f"sequence lengths are {expected_end}. This indicates a malformed CIGAR from "
            f"the aligner, not a data issue -- investigate before trusting any output "
            f"from this run."
        )

    return noisy_trimmed, clean_trimmed, breakpoints


def simulate_training_data(cfg: SimulatorConfig) -> dict:
    """
    Full pipeline: reference FASTA -> Badread simulated reads -> minimap2
    alignment -> breakpoints -> JSONL manifest at cfg.output_jsonl, in the
    exact schema data/dataset.py.load_aligned_pairs_from_jsonl() expects.

    Returns a stats dict (reads_simulated, pairs_written, reads_skipped)
    for logging/reporting.
    """
    header, reference_seq = load_reference(cfg.reference_fasta)
    print(f"Loaded reference '{header}' ({len(reference_seq):,} bp) from {cfg.reference_fasta}")

    print(f"Running Badread (quantity={cfg.quantity}, seed={cfg.seed})...")
    fastq_text = run_badread(cfg.reference_fasta, cfg.quantity, cfg.seed, cfg.identity, cfg.badread_extra_args)
    reads = parse_fastq(fastq_text)
    print(f"Badread produced {len(reads)} simulated read(s).")

    print("Indexing reference for alignment (minimap2 map-ont preset)...")
    aligner = mp.Aligner(cfg.reference_fasta, preset="map-ont")
    if not aligner:
        raise RuntimeError(f"mappy failed to index reference {cfg.reference_fasta}")

    written = 0
    Path(cfg.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_jsonl, "w") as out_f:
        for read_header, read_seq in reads:
            result = align_read_to_reference(aligner, read_seq, reference_seq, cfg.min_aligned_length)
            if result is None:
                continue
            noisy_trimmed, clean_trimmed, breakpoints = result
            record = {
                "noisy_sequence": noisy_trimmed,
                "clean_sequence": clean_trimmed,
                "breakpoints": breakpoints,
                "source_read_header": read_header,  # provenance only -- not read by dataset.py, useful for debugging
            }
            out_f.write(json.dumps(record) + "\n")
            written += 1

    stats = {
        "reads_simulated": len(reads),
        "pairs_written": written,
        "reads_skipped": len(reads) - written,
    }
    print(
        f"Wrote {written} aligned training pair(s) to {cfg.output_jsonl} "
        f"({stats['reads_skipped']} read(s) skipped: no usable primary alignment, or aligned "
        f"span below min_aligned_length={cfg.min_aligned_length})."
    )
    return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate ONT training pairs via Badread + minimap2.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE_FASTA_PATH, help="Path to reference FASTA (single record).")
    parser.add_argument("--output", default=DEFAULT_TRAINING_DATA_JSONL_PATH, help="Path to write the output JSONL manifest.")
    parser.add_argument("--quantity", default="50x", help="Badread --quantity, e.g. '50x' or an absolute bp count.")
    parser.add_argument("--identity", default=None, help="Badread --identity 'mean,max,stdev'; omit for Badread's default ONT profile.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-aligned-length", type=int, default=50)
    return parser


if __name__ == "__main__":
    import sys

    # If invoked with CLI args (real usage), run the real pipeline against
    # whatever reference/output paths were given and exit.
    if len(sys.argv) > 1:
        args = _build_arg_parser().parse_args()
        cfg = SimulatorConfig(
            reference_fasta=args.reference,
            output_jsonl=args.output,
            quantity=args.quantity,
            identity=args.identity,
            seed=args.seed,
            min_aligned_length=args.min_aligned_length,
        )
        simulate_training_data(cfg)
        sys.exit(0)

    # ------------------------------------------------------------------
    # No CLI args: run the sanity-check suite instead.
    # ------------------------------------------------------------------
    import tempfile

    from config import PAD_IDX
    from data.dataset import (
        AlignedPair,
        GenomeCorrectionDataset,
        create_dataloader,
        load_aligned_pairs_from_jsonl,
    )
    from model.sequence_translation_model import SequenceTranslationConfig, SequenceTranslationModel
    import torch
    import torch.nn as nn

    # -- 1. load_reference() against the REAL, full 4.6 Mb genome -------------
    real_header, real_seq = load_reference(DEFAULT_REFERENCE_FASTA_PATH)
    assert "MG1655" in real_header or "K-12" in real_header, real_header
    assert 4_600_000 < len(real_seq) < 4_700_000, len(real_seq)
    assert set(real_seq) <= set("ACGTN")
    print(f"[1/4] load_reference() verified against the real genome: '{real_header}' ({len(real_seq):,} bp).")

    # -- 2. Build a small excerpt for a FAST end-to-end run (Badread + -------
    #       minimap2 on the full 4.6 Mb genome works identically, just slower;
    #       the excerpt exercises the exact same code path). -----------------
    with tempfile.TemporaryDirectory() as tmpdir:
        excerpt_path = str(Path(tmpdir) / "ecoli_excerpt.fasta")
        excerpt_seq = real_seq[100_000:150_000]  # 50 kb, real genomic sequence, just a smaller slice
        with open(excerpt_path, "w") as f:
            f.write(">ecoli_excerpt\n")
            for i in range(0, len(excerpt_seq), 70):
                f.write(excerpt_seq[i:i + 70] + "\n")

        output_jsonl_path = str(Path(tmpdir) / "training_pairs.jsonl")
        cfg = SimulatorConfig(
            reference_fasta=excerpt_path,
            output_jsonl=output_jsonl_path,
            quantity="5x",
            seed=42,
        )
        stats = simulate_training_data(cfg)
        assert stats["pairs_written"] > 0, "Expected at least one usable aligned pair from the smoke test"
        print(f"[2/4] Full Badread -> minimap2 -> JSONL pipeline passed. Stats: {stats}")

        # -- 3. Load the JSONL back and validate every AlignedPair ------------
        pairs = load_aligned_pairs_from_jsonl(output_jsonl_path)
        assert len(pairs) == stats["pairs_written"]
        for p in pairs:
            p.validate()  # raises if any breakpoint contract is violated
        print(f"[3/4] load_aligned_pairs_from_jsonl() + AlignedPair.validate() passed for all {len(pairs)} pair(s).")

        # -- 4. Full-stack integration: real reads -> real alignment -> ------
        #       real Dataset/collate_fn -> real model -> real backward pass ---
        torch.manual_seed(0)
        loader = create_dataloader(pairs, chunk_size=512, min_chunk_size=8, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        model = SequenceTranslationModel(SequenceTranslationConfig())
        output = model(
            src_tokens=batch["src_tokens"],
            src_lengths=batch["src_lengths"],
            rle_base_ids=batch["rle_base_ids"],
            rle_run_lengths=batch["rle_run_lengths"],
            target_tokens=batch["target_tokens"],
            teacher_forcing_ratio=1.0,
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        targets_for_loss = batch["target_tokens"][:, 1:]
        loss = loss_fn(output.logits.reshape(-1, output.logits.size(-1)), targets_for_loss.reshape(-1))
        loss.backward()
        assert torch.isfinite(loss)
        assert model.encoder.embedding.weight.grad is not None
        print(
            f"[4/4] Full-stack integration passed on REAL Badread reads from the REAL E. coli "
            f"genome, REAL minimap2 alignment, through GenomeCorrectionDataset -> "
            f"SequenceTranslationModel -> backward(). Loss = {loss.item():.4f}"
        )

    print("\nAll simulator sanity checks passed.")