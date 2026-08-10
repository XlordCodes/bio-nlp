"""
training/variant_preservation_test.py
-----------------------------------------
Tests the SPECIFIC failure mode research flagged for single-read (non-MSA)
correctors: "reference hallucination" -- the model memorizing the training
genome's k-mer distribution strongly enough that it overwrites a genuine
rare variant with the canonical base it saw most often during training,
mistaking true biology for sequencing noise.

This is a DIFFERENT risk than the "variant erasure via majority vote" that
affects consensus/MSA tools like pre-HERRO polishers -- our architecture is
structurally immune to THAT specific failure (it never sees other reads to
vote against). This test targets the risk that IS relevant to us.

METHODOLOGY
  1. Take a reference chunk. Plant N random point mutations at known
     positions -- this becomes the "true" sequence for a hypothetical
     sample that genuinely differs from the training-time reference at
     these specific loci (simulating a real strain-level SNP).
  2. Simulate a noisy ONT read FROM THE MUTATED SEQUENCE (not the original
     reference) via Badread -- the read's true biological origin is the
     mutated sequence, exactly as a real sequencing read's origin would be
     whatever DNA molecule was actually in the sample, not a reference genome.
  3. Run the model's correction on the noisy read.
  4. For each planted mutation, align the corrected output against the
     mutated ("true") sequence via edlib and classify the model's output at
     that specific position:
       - PRESERVED : matches the true mutated base (correct)
       - ERASED    : reverted to the ORIGINAL reference base (hallucination
                      -- the specific failure this test exists to catch)
       - OTHER     : matches neither (an ordinary correction error, not the
                      specific hallucination failure mode)
       - UNALIGNED : the position couldn't be resolved in the alignment
                      (reported separately, not counted in the rate below)
  5. Report variant_preservation_rate = PRESERVED / (PRESERVED + ERASED).
     OTHER and UNALIGNED are reported but excluded from this rate, since
     they're not evidence of hallucination specifically.

REQUIRES A TRAINED CHECKPOINT to be meaningful. On an untrained model, all
outcomes are noise -- this is a post-training validation tool, not a
build-time sanity check like this project's other __main__ test blocks.

USAGE (run on your own machine, after training -- see docs/BENCHMARKING.md):
    python -m training.variant_preservation_test \\
        --checkpoint checkpoints/model_best.pt \\
        --reference data/reference/ecoli_k12_mg1655.fasta \\
        --num-chunks 50 --mutations-per-chunk 3
"""

import argparse
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import edlib
import torch

from config import DEFAULT_MODEL_CHECKPOINT_PATH, DEFAULT_REFERENCE_FASTA_PATH
from backend.inference_engine import InferenceEngine


@dataclass
class PlantedMutation:
    position: int  # 0-indexed position within the reference chunk
    original_base: str
    mutated_base: str


def load_reference_sequence(fasta_path: str) -> str:
    """Minimal single-record FASTA loader (mirrors data/simulator.py's, kept
    independent here so this script has no import-time dependency on it)."""
    seq_parts = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                seq_parts.append(line.upper())
    return "".join(seq_parts)


def plant_mutations(
    reference_chunk: str, num_mutations: int, rng: random.Random
) -> Tuple[str, List[PlantedMutation]]:
    """Returns (mutated_sequence, planted mutations) for one reference chunk."""
    if num_mutations > len(reference_chunk):
        raise ValueError(
            f"Cannot plant {num_mutations} mutations into a {len(reference_chunk)}-base chunk."
        )
    positions = sorted(rng.sample(range(len(reference_chunk)), num_mutations))
    mutated = list(reference_chunk)
    mutations = []
    for pos in positions:
        original = mutated[pos]
        alt = rng.choice([b for b in "ACGT" if b != original])
        mutations.append(PlantedMutation(position=pos, original_base=original, mutated_base=alt))
        mutated[pos] = alt
    return "".join(mutated), mutations


def classify_outcome(corrected_seq: str, true_seq: str, mutation: PlantedMutation) -> str:
    """
    Aligns corrected_seq against true_seq to find what the model actually
    output at the planted mutation's position, then classifies it as
    PRESERVED / ERASED / OTHER / UNALIGNED. See module docstring.
    """
    result = edlib.align(true_seq, corrected_seq, mode="NW", task="path")
    nice = edlib.getNiceAlignment(result, true_seq, corrected_seq)
    true_aligned, corrected_aligned = nice["query_aligned"], nice["target_aligned"]

    true_pos = 0
    corrected_char = None
    for t_char, c_char in zip(true_aligned, corrected_aligned):
        if t_char != "-":
            if true_pos == mutation.position:
                corrected_char = c_char
                break
            true_pos += 1

    if corrected_char is None or corrected_char == "-":
        return "UNALIGNED"
    if corrected_char == mutation.mutated_base:
        return "PRESERVED"
    if corrected_char == mutation.original_base:
        return "ERASED"
    return "OTHER"


def run_variant_preservation_test(
    checkpoint_path: str,
    reference_fasta_path: str,
    chunk_length: int = 300,
    num_chunks: int = 20,
    mutations_per_chunk: int = 3,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)

    print(f"Loading checkpoint from {checkpoint_path} ...")
    engine = InferenceEngine.load_from_checkpoint(checkpoint_path, device=device)

    reference = load_reference_sequence(reference_fasta_path)
    print(f"Loaded reference ({len(reference):,} bp) from {reference_fasta_path}")

    outcomes = {"PRESERVED": 0, "ERASED": 0, "OTHER": 0, "UNALIGNED": 0}
    per_mutation_records = []

    for chunk_idx in range(num_chunks):
        start = rng.randint(0, len(reference) - chunk_length)
        reference_chunk = reference[start : start + chunk_length]

        true_chunk, mutations = plant_mutations(reference_chunk, mutations_per_chunk, rng)

        result = engine.correct_sequence(true_chunk)
        # NOTE: this deliberately feeds the (clean) mutated sequence straight
        # to correction without an intermediate Badread noise pass, to
        # isolate whether the MODEL ITSELF reverts a true variant it has no
        # noise-related reason to distrust. If you also want to test
        # preservation THROUGH realistic noise, run Badread on true_chunk
        # first and correct the noisy output instead -- see
        # docs/BENCHMARKING.md for both variants of this test.
        corrected = result["corrected_sequence"]

        for mutation in mutations:
            outcome = classify_outcome(corrected, true_chunk, mutation)
            outcomes[outcome] += 1
            per_mutation_records.append(
                {
                    "chunk_index": chunk_idx,
                    "position_in_chunk": mutation.position,
                    "original_base": mutation.original_base,
                    "mutated_base": mutation.mutated_base,
                    "outcome": outcome,
                }
            )

        print(
            f"[{chunk_idx + 1}/{num_chunks}] chunk processed, "
            f"{len(mutations)} mutation(s), running tally: {outcomes}"
        )

    denominator = outcomes["PRESERVED"] + outcomes["ERASED"]
    preservation_rate = outcomes["PRESERVED"] / denominator if denominator > 0 else float("nan")

    summary = {
        "num_chunks": num_chunks,
        "mutations_per_chunk": mutations_per_chunk,
        "total_mutations_tested": sum(outcomes.values()),
        "outcomes": outcomes,
        "variant_preservation_rate": preservation_rate,
    }

    print("\n=== Variant Preservation Test Summary ===")
    print(f"Total mutations tested: {summary['total_mutations_tested']}")
    print(f"Outcomes: {outcomes}")
    print(f"Variant preservation rate (PRESERVED / (PRESERVED + ERASED)): {preservation_rate:.4f}")
    print(
        "\nInterpretation: closer to 1.0 means the model correctly trusts genuine variants; "
        "closer to 0.0 means it's reverting true mutations back to the reference it trained on "
        "-- the 'reference hallucination' failure mode this test targets. OTHER/UNALIGNED "
        "counts are ordinary correction errors, not hallucination evidence specifically."
    )

    return {"summary": summary, "per_mutation": per_mutation_records}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test variant preservation vs. reference hallucination.")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL_CHECKPOINT_PATH)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE_FASTA_PATH)
    parser.add_argument("--chunk-length", type=int, default=300)
    parser.add_argument("--num-chunks", type=int, default=20)
    parser.add_argument("--mutations-per-chunk", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_variant_preservation_test(
        checkpoint_path=args.checkpoint,
        reference_fasta_path=args.reference,
        chunk_length=args.chunk_length,
        num_chunks=args.num_chunks,
        mutations_per_chunk=args.mutations_per_chunk,
        seed=args.seed,
    )
