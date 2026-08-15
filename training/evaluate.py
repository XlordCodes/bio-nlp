"""
training/evaluate.py
----------------------
Standalone evaluation script. Unlike training/train.py's cheap per-epoch
validation loss (teacher-forced, used only to pick a checkpoint), this file
runs the model FREE-RUNNING (no teacher forcing -- exactly how it behaves
at real inference time) via the exact same InferenceEngine used in
production (backend/inference_engine.py), against known ground truth, and
reports the full suite of bioinformatics metrics from training/metrics.py:
edit distance/identity, reading-frame preservation, and external minimap2
alignment verification. Results are also dumped to FASTA for external
inspection, per the Part 3 spec.

Reusing InferenceEngine.correct_sequence() here (rather than calling
model.predict() directly) is deliberate: it means evaluation numbers
reflect the actual chunking + stitching pipeline real requests go through,
not an idealized single-shot decode that would never be representative of
production behavior on long reads.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import List, Optional

import torch

from config import (
    DEFAULT_MODEL_CHECKPOINT_PATH, DEFAULT_INFERENCE_CHUNK_SIZE,
    DEFAULT_INFERENCE_CHUNK_OVERLAP, DEFAULT_DECODE_LENGTH_MARGIN,
)
from data.dataset import AlignedPair, load_aligned_pairs_from_jsonl
from backend.inference_engine import InferenceEngine
from training.metrics import compute_all_metrics


def evaluate_pair(engine: InferenceEngine, pair: AlignedPair, run_minimap2: bool) -> dict:
    """Runs one (noisy, clean) pair through the real inference pipeline and computes all metrics against clean_sequence as ground truth."""
    correction_result = engine.correct_sequence(pair.noisy_sequence)
    predicted_seq = correction_result["corrected_sequence"]

    metrics = compute_all_metrics(predicted_seq, pair.clean_sequence, run_minimap2=run_minimap2)
    metrics["predicted_sequence"] = predicted_seq
    metrics["operational_metrics"] = correction_result["metrics"]  # inference-time metrics (vs input, not ground truth)
    return metrics


def run_evaluation(
    checkpoint_path: str,
    validation_jsonl_path: str,
    output_dir: str,
    max_examples: Optional[int] = None,
    run_minimap2: bool = True,
    chunk_size: int = DEFAULT_INFERENCE_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_INFERENCE_CHUNK_OVERLAP,
    decode_length_margin: int = DEFAULT_DECODE_LENGTH_MARGIN,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Full evaluation run. Loads a checkpoint, evaluates every (noisy, clean)
    pair in validation_jsonl_path (or the first max_examples of them),
    writes a FASTA dump and a JSON summary to output_dir, and returns the
    aggregate report dict.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path} ...")
    engine = InferenceEngine.load_from_checkpoint(
        checkpoint_path, device=device, chunk_size=chunk_size, chunk_overlap=chunk_overlap, decode_length_margin=decode_length_margin
    )

    pairs = load_aligned_pairs_from_jsonl(validation_jsonl_path)
    if max_examples is not None:
        pairs = pairs[:max_examples]
    print(f"Evaluating {len(pairs)} pair(s) from {validation_jsonl_path} ...")

    per_example_results = []
    fasta_path = output_dir_path / "evaluation_outputs.fasta"
    with open(fasta_path, "w") as fasta_f:
        for i, pair in enumerate(pairs):
            metrics = evaluate_pair(engine, pair, run_minimap2=run_minimap2)
            per_example_results.append(metrics)

            fasta_f.write(f">example_{i}_reference\n{pair.clean_sequence}\n")
            fasta_f.write(f">example_{i}_noisy_input\n{pair.noisy_sequence}\n")
            fasta_f.write(f">example_{i}_predicted\n{metrics['predicted_sequence']}\n")

            print(
                f"[{i+1}/{len(pairs)}] identity={metrics['alignment']['identity']:.3f} "
                f"edit_distance={metrics['alignment']['edit_distance']} "
                f"frame_preserved={metrics['frame']['global_frame_preserved']}"
            )

    # -- aggregate ------------------------------------------------------------
    identities = [r["alignment"]["identity"] for r in per_example_results]
    edit_distances = [r["alignment"]["edit_distance"] for r in per_example_results]
    frame_preserved_flags = [r["frame"]["global_frame_preserved"] for r in per_example_results]
    frame_intact_fractions = [r["frame"]["frame_intact_fraction"] for r in per_example_results]

    # How often the decoder failed to terminate naturally (no <EOS> within
    # its decode budget) across the whole run -- a genuine free-running
    # quality signal (see InferenceEngine.correct_sequence(), which tracks
    # this per chunk instead of printing a warning per occurrence). Chunks
    # that don't terminate naturally tend to drift/degrade for their
    # remaining budget, which inflates edit_distance/depresses identity for
    # those specific examples -- report the rate so it's visible as a real
    # number instead of buried in per-example noise.
    total_chunks = sum(r["operational_metrics"]["num_chunks"] for r in per_example_results)
    total_chunks_without_eos = sum(r["operational_metrics"]["chunks_without_eos"] for r in per_example_results)
    examples_with_any_truncation = sum(
        1 for r in per_example_results if r["operational_metrics"]["chunks_without_eos"] > 0
    )

    summary = {
        "num_examples": len(per_example_results),
        "mean_identity": statistics.mean(identities) if identities else float("nan"),
        "mean_edit_distance": statistics.mean(edit_distances) if edit_distances else float("nan"),
        "fraction_frame_preserved": (
            sum(frame_preserved_flags) / len(frame_preserved_flags) if frame_preserved_flags else float("nan")
        ),
        "mean_frame_intact_fraction": (
            statistics.mean(frame_intact_fractions) if frame_intact_fractions else float("nan")
        ),
        "total_chunks": total_chunks,
        "total_chunks_without_eos": total_chunks_without_eos,
        "fraction_chunks_without_eos": (
            total_chunks_without_eos / total_chunks if total_chunks > 0 else float("nan")
        ),
        "fraction_examples_with_any_truncation": (
            examples_with_any_truncation / len(per_example_results) if per_example_results else float("nan")
        ),
    }

    if run_minimap2:
        mm2_identities = [
            r["minimap2"]["identity"] for r in per_example_results
            if r.get("minimap2", {}).get("aligned")
        ]
        mm2_coverages = [
            r["minimap2"]["coverage"] for r in per_example_results
            if r.get("minimap2", {}).get("aligned")
        ]
        summary["minimap2_fraction_aligned"] = (
            len(mm2_identities) / len(per_example_results) if per_example_results else float("nan")
        )
        summary["minimap2_mean_identity"] = statistics.mean(mm2_identities) if mm2_identities else float("nan")
        summary["minimap2_mean_coverage"] = statistics.mean(mm2_coverages) if mm2_coverages else float("nan")

    summary_path = output_dir_path / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "per_example": per_example_results}, f, indent=2)

    print(f"\nWrote FASTA dump to {fasta_path}")
    print(f"Wrote summary JSON to {summary_path}")
    print(f"\n=== Summary over {summary['num_examples']} example(s) ===")
    for k, v in summary.items():
        if k != "num_examples":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if summary["fraction_chunks_without_eos"] > 0.05:
        print(
            f"\nNOTE: {summary['fraction_chunks_without_eos']:.1%} of chunks across this run did not "
            f"terminate naturally (no <EOS> within decode budget). This is a genuine free-running-quality "
            f"signal, not just log noise -- affected examples' identity/edit_distance are likely pessimistic "
            f"underestimates of true correction quality, since a truncated-without-stopping tail tends to "
            f"drift/degrade rather than cleanly end. Common on an early-training checkpoint (val_loss is "
            f"measured under full teacher forcing and never scores the model on deciding when to stop); "
            f"expect this fraction to drop as training progresses further."
        )

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint against ground truth.")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL_CHECKPOINT_PATH)
    parser.add_argument("--validation-data", required=False, help="JSONL of AlignedPairs to evaluate against.")
    parser.add_argument("--output-dir", default="evaluation_output")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--decode-length-margin", type=int, default=DEFAULT_DECODE_LENGTH_MARGIN)
    parser.add_argument("--no-minimap2", action="store_true", help="Skip the external minimap2 verification step.")
    return parser


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        args = _build_arg_parser().parse_args()
        if not args.validation_data:
            print("--validation-data is required for a real evaluation run.")
            sys.exit(1)
        run_evaluation(
            checkpoint_path=args.checkpoint,
            validation_jsonl_path=args.validation_data,
            output_dir=args.output_dir,
            max_examples=args.max_examples,
            run_minimap2=not args.no_minimap2,
            decode_length_margin=args.decode_length_margin,
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # No CLI args: run a small, fast, REAL end-to-end evaluation sanity
    # check -- trains a tiny model for a couple of epochs on real data
    # (reusing training/train.py's own smoke-test-scale approach), then
    # evaluates the resulting real checkpoint on a held-out real pair,
    # verifying the full pipeline (checkpoint load -> InferenceEngine ->
    # chunking/stitching -> metrics -> FASTA dump) runs correctly.
    # ------------------------------------------------------------------
    import tempfile

    from data.dataset import _chunk_aligned_pair
    from data.simulator import SimulatorConfig, simulate_training_data
    from training.train import train

    def _truncate_pair(pair: AlignedPair, max_len: int) -> AlignedPair:
        noisy_sub, clean_sub = _chunk_aligned_pair(pair, chunk_size=max_len, min_chunk_size=1)[0]
        return AlignedPair(noisy_sub, clean_sub, breakpoints=[(0, 0), (len(noisy_sub), len(clean_sub))])

    with tempfile.TemporaryDirectory() as tmpdir:
        with open("data/reference/ecoli_k12_mg1655.fasta") as f:
            lines = f.readlines()
        real_seq = "".join(l.strip() for l in lines[1:])
        excerpt_seq = real_seq[450_000:453_000]  # a different real 3kb excerpt from train.py's smoke test

        excerpt_path = str(Path(tmpdir) / "excerpt.fasta")
        with open(excerpt_path, "w") as f:
            f.write(">excerpt\n")
            for i in range(0, len(excerpt_seq), 70):
                f.write(excerpt_seq[i:i + 70] + "\n")

        raw_jsonl_path = str(Path(tmpdir) / "raw_pairs.jsonl")
        stats = simulate_training_data(
            SimulatorConfig(reference_fasta=excerpt_path, output_jsonl=raw_jsonl_path, quantity="10x", seed=11)
        )
        assert stats["pairs_written"] >= 6

        raw_pairs = load_aligned_pairs_from_jsonl(raw_jsonl_path)
        small_pairs = [_truncate_pair(p, max_len=100) for p in raw_pairs[:6]]

        train_jsonl_path = str(Path(tmpdir) / "train_pairs.jsonl")
        with open(train_jsonl_path, "w") as f:
            for p in small_pairs:
                f.write(json.dumps({
                    "noisy_sequence": p.noisy_sequence, "clean_sequence": p.clean_sequence,
                    "breakpoints": p.breakpoints,
                }) + "\n")

        checkpoint_path = str(Path(tmpdir) / "checkpoint.pt")
        train(
            train_jsonl_path=train_jsonl_path,
            val_fraction=0.3,
            num_epochs=3,
            batch_size=4,
            learning_rate=2e-3,
            chunk_size=150,
            min_chunk_size=1,
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu"),
            seed=11,
            log_every=1000,
        )
        assert Path(checkpoint_path).exists()
        print(f"\n[setup] Trained a real tiny checkpoint at {checkpoint_path} for the evaluation smoke test.")

        output_dir = str(Path(tmpdir) / "eval_output")
        summary = run_evaluation(
            checkpoint_path=checkpoint_path,
            validation_jsonl_path=train_jsonl_path,  # reusing the same tiny pairs is fine for a plumbing test
            output_dir=output_dir,
            max_examples=3,
            run_minimap2=True,
            chunk_size=150,
            chunk_overlap=32,
            decode_length_margin=DEFAULT_DECODE_LENGTH_MARGIN,
            device=torch.device("cpu"),
        )

        assert summary["num_examples"] == 3
        assert 0.0 <= summary["mean_identity"] <= 1.0
        assert 0.0 <= summary["fraction_frame_preserved"] <= 1.0
        assert 0.0 <= summary["fraction_chunks_without_eos"] <= 1.0
        print(f"\n[1/3] run_evaluation() produced a well-formed summary: {summary}")

        fasta_path = Path(output_dir) / "evaluation_outputs.fasta"
        assert fasta_path.exists()
        fasta_content = fasta_path.read_text()
        assert fasta_content.count(">") == 9  # 3 examples x 3 records (reference/noisy/predicted)
        print(f"[2/3] FASTA dump written correctly: {fasta_path} ({fasta_content.count('>')} records).")

        summary_json_path = Path(output_dir) / "evaluation_summary.json"
        assert summary_json_path.exists()
        with open(summary_json_path) as f:
            loaded = json.load(f)
        assert "summary" in loaded and "per_example" in loaded
        assert len(loaded["per_example"]) == 3
        print(f"[3/3] JSON summary written and re-loadable: {summary_json_path}")

    print("\nAll evaluate.py sanity checks passed.")