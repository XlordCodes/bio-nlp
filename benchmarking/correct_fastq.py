"""
benchmarking/correct_fastq.py
--------------------------------
Batch-corrects every read in a FASTQ file through a trained checkpoint,
writing a corrected FASTA (quality scores aren't meaningful post-correction,
so output is FASTA not FASTQ) -- the missing piece needed to run "our
model's corrected reads" through the same assembly pipeline (Flye) as the
other baselines in benchmarking/run_comparison.sh.

RUN THIS ON YOUR OWN MACHINE, NOT IN THE SANDBOX -- it needs a real trained
checkpoint, which doesn't exist yet (see docs/BENCHMARKING.md).

Usage:
    python -m benchmarking.correct_fastq \\
        --checkpoint checkpoints/model_best.pt \\
        --input benchmark_data/ecoli_subset/subset_50x.fastq \\
        --output benchmark_data/ecoli_subset/corrected_by_ours.fasta
"""

import argparse
import time
from pathlib import Path

import torch

from config import DEFAULT_MODEL_CHECKPOINT_PATH
from backend.inference_engine import InferenceEngine


def parse_fastq(path: str):
    """Minimal FASTQ reader -- yields (header, sequence) pairs, ignoring quality strings."""
    with open(path) as f:
        while True:
            header = f.readline().rstrip()
            if not header:
                break
            seq = f.readline().rstrip()
            f.readline()  # '+' separator line
            f.readline()  # quality string, unused
            yield header.lstrip("@"), seq


def correct_fastq(checkpoint_path: str, input_path: str, output_path: str, device=None, log_every: int = 50) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint_path} on device={device} ...")
    engine = InferenceEngine.load_from_checkpoint(checkpoint_path, device=device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_reads = 0
    total_bases_in = 0
    total_bases_out = 0
    failed_reads = 0
    start_time = time.perf_counter()

    with open(output_path, "w") as out_f:
        for header, seq in parse_fastq(input_path):
            total_reads += 1
            if total_reads % log_every == 0:
                elapsed = time.perf_counter() - start_time
                rate = total_bases_in / elapsed / 1_000_000 if elapsed > 0 else 0
                print(f"  {total_reads} reads processed ({rate:.3f} Mb/s so far) ...")

            if len(seq) == 0:
                failed_reads += 1
                continue

            try:
                result = engine.correct_sequence(seq)
            except Exception as e:
                print(f"  WARNING: correction failed for read '{header}': {e}")
                failed_reads += 1
                continue

            corrected = result["corrected_sequence"]
            total_bases_in += len(seq)
            total_bases_out += len(corrected)
            out_f.write(f">{header}\n{corrected}\n")

    elapsed = time.perf_counter() - start_time
    stats = {
        "total_reads": total_reads,
        "successfully_corrected": total_reads - failed_reads,
        "failed_reads": failed_reads,
        "total_bases_in": total_bases_in,
        "total_bases_out": total_bases_out,
        "elapsed_seconds": elapsed,
        "throughput_mb_per_sec": (total_bases_in / elapsed / 1_000_000) if elapsed > 0 else 0,
    }

    print("\n=== correct_fastq.py summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nCorrected reads written to {output_path}")
    print(
        "\nNOTE ON THROUGHPUT: report this Mb/s figure alongside the exact hardware used "
        "(CPU model or GPU model + VRAM) when comparing against Medaka/Racon/dorado correct "
        "throughput figures -- see docs/BENCHMARKING.md for the standard reporting format "
        "this project follows."
    )
    return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-correct a FASTQ file through a trained checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL_CHECKPOINT_PATH)
    parser.add_argument("--input", required=True, help="Input FASTQ file of noisy reads.")
    parser.add_argument("--output", required=True, help="Output FASTA file of corrected reads.")
    parser.add_argument("--log-every", type=int, default=50)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    correct_fastq(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
        log_every=args.log_every,
    )
