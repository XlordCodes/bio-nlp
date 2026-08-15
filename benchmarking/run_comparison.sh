#!/usr/bin/env bash
# benchmarking/run_comparison.sh
# ----------------------------------
# Runs the standard baseline pipelines alongside our model's correction on
# the same reads, then assesses every output against the target reference
# using pomoxis (assess_assembly, assess_homopolymers) and dnadiff --
# matching the exact methodology published Medaka/Racon benchmarks use, so
# our numbers are directly comparable to the literature, not just internally
# consistent.
#
# RUN THIS ON YOUR OWN MACHINE, NOT IN THE SANDBOX. Requires: flye, racon,
# medaka, pomoxis, dnadiff (MUMmer4), and our own checkpoint. See
# docs/BENCHMARKING.md for install commands for every tool referenced here.
#
# dorado correct / HERRO is deliberately NOT included in this script --
# per research it needs data-center-class GPU resources (32GB+ VRAM,
# 64+ CPU cores recommended) to be practical even for a single bacterial
# genome. Run it manually and separately if you have access to that
# hardware; see docs/BENCHMARKING.md for the exact invocation.
#
# Usage:
#   ./run_comparison.sh <reads.fastq> <target_reference.fasta> <our_checkpoint.pt> <output_dir> [genome_size] [medaka_model]

set -euo pipefail

READS="${1:?Usage: $0 <reads.fastq> <target_reference.fasta> <our_checkpoint.pt> <output_dir>}"
REFERENCE="${2:?Provide the target reference FASTA -- must match the reads true strain, see config.py}"
CHECKPOINT="${3:?Provide our trained model checkpoint path}"
OUTPUT_DIR="${4:?Provide an output directory}"
GENOME_SIZE="${5:-4.8m}"
MEDAKA_MODEL="${6:-}"  # optional; run `medaka tools list_models` to pick one matching your
                        # read chemistry (e.g. R10.4.1 sup). Empty = medaka_consensus's bundled default.

# --- logging: mirror everything below to a timestamped file, not just the terminal ---
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run_comparison_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== run_comparison.sh started: $(date) ==="
echo "Logging to: $LOG_FILE"

for tool in flye racon medaka_consensus assess_assembly assess_homopolymers dnadiff minimap2; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' not found on PATH. See docs/BENCHMARKING.md."; exit 1; }
done

mkdir -p "$OUTPUT_DIR"/{uncorrected,racon_medaka,ours,assessments}
THREADS="$(nproc)"

# ---------------------------------------------------------------------------
# Pipeline A: Uncorrected baseline -- assemble raw reads directly, no polishing
# ---------------------------------------------------------------------------
echo "=== [A] Uncorrected baseline: Flye on raw reads ==="
flye --nano-hq "$READS" --out-dir "$OUTPUT_DIR/uncorrected" --threads "$THREADS" -g "$GENOME_SIZE"

# ---------------------------------------------------------------------------
# Pipeline B: Standard industry pipeline -- Flye -> Racon (x4) -> Medaka
# ---------------------------------------------------------------------------
echo "=== [B] Standard pipeline: Flye -> Racon x4 -> Medaka ==="
flye --nano-hq "$READS" --out-dir "$OUTPUT_DIR/racon_medaka/flye_draft" --threads "$THREADS" -g "$GENOME_SIZE"
DRAFT="$OUTPUT_DIR/racon_medaka/flye_draft/assembly.fasta"

CURRENT="$DRAFT"
for i in 1 2 3 4; do
    minimap2 -ax map-ont -t "$THREADS" "$CURRENT" "$READS" > "$OUTPUT_DIR/racon_medaka/round${i}.sam"
    racon -t "$THREADS" "$READS" "$OUTPUT_DIR/racon_medaka/round${i}.sam" "$CURRENT" \
        > "$OUTPUT_DIR/racon_medaka/racon_round${i}.fasta"
    CURRENT="$OUTPUT_DIR/racon_medaka/racon_round${i}.fasta"
done

if [ -n "$MEDAKA_MODEL" ]; then
    medaka_consensus -i "$READS" -d "$CURRENT" -o "$OUTPUT_DIR/racon_medaka/medaka_out" -t "$THREADS" -m "$MEDAKA_MODEL"
else
    medaka_consensus -i "$READS" -d "$CURRENT" -o "$OUTPUT_DIR/racon_medaka/medaka_out" -t "$THREADS"
fi
RACON_MEDAKA_FINAL="$OUTPUT_DIR/racon_medaka/medaka_out/consensus.fasta"

# ---------------------------------------------------------------------------
# Pipeline C: Our model -- correct reads first, then assemble the (already
# corrected) reads with Flye, WITHOUT any subsequent Racon/Medaka polishing.
# This is the specific comparison that determines whether pre-assembly
# single-read correction can genuinely replace post-assembly polishing --
# see docs/BENCHMARKING.md's success-criteria section.
# ---------------------------------------------------------------------------
echo "=== [C] Our model: correct reads, then Flye, no post-assembly polishing ==="
python -m benchmarking.correct_fastq \
    --checkpoint "$CHECKPOINT" \
    --input "$READS" \
    --output "$OUTPUT_DIR/ours/corrected_reads.fasta"

flye --nano-hq "$OUTPUT_DIR/ours/corrected_reads.fasta" --out-dir "$OUTPUT_DIR/ours/flye_out" --threads "$THREADS" -g "$GENOME_SIZE"
OURS_FINAL="$OUTPUT_DIR/ours/flye_out/assembly.fasta"

# ---------------------------------------------------------------------------
# Assessment: pomoxis + dnadiff for all three, against the SAME reference
# ---------------------------------------------------------------------------
echo "=== Assessing all three pipelines against $REFERENCE ==="

assess_one() {
    local name="$1"
    local assembly="$2"
    echo "--- assess_assembly: $name ---"
    assess_assembly -r "$REFERENCE" -i "$assembly" -p "$OUTPUT_DIR/assessments/${name}_assess"
    echo "--- assess_homopolymers: $name (this is the number that matters most for our RLE-channel claim) ---"
    assess_homopolymers -r "$REFERENCE" -i "$assembly" -p "$OUTPUT_DIR/assessments/${name}_homopolymers"
    echo "--- dnadiff: $name ---"
    dnadiff -p "$OUTPUT_DIR/assessments/${name}_dnadiff" "$REFERENCE" "$assembly"
}

assess_one "uncorrected" "$OUTPUT_DIR/uncorrected/assembly.fasta"
assess_one "racon_medaka" "$RACON_MEDAKA_FINAL"
assess_one "ours" "$OURS_FINAL"

echo ""
echo "=== Done. Reports are in $OUTPUT_DIR/assessments/ ==="
echo "Compare each pipeline's *_assess.summary (Q-scores, indel/mismatch breakdown)"
echo "and *_homopolymers output (homopolymer-specific accuracy -- the RLE-channel claim)."
echo ""
echo "=== Full log saved to: $LOG_FILE ==="