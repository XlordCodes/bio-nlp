#!/usr/bin/env bash
# benchmarking/prepare_zymo_subset.sh
# --------------------------------------
# Extracts a single-species, coverage-normalized read subset from raw Zymo
# mock-community sequencing data -- standard practice for a compute-
# constrained proof-of-concept rather than processing the full multi-
# gigabase metagenomic community (per research: map to target species with
# minimap2, extract mapped reads, downsample to normalized depth with
# Rasusa).
#
# RUN THIS ON YOUR OWN MACHINE, NOT IN THE SANDBOX. Requires: minimap2,
# samtools, rasusa on PATH. See docs/BENCHMARKING.md for install commands.
#
# Usage:
#   ./prepare_zymo_subset.sh <raw_zymo_reads.fastq.gz> <target_reference.fasta> <output_dir> [target_coverage] [genome_size]
#
# Example (E. coli from Zymo D6300, targeting 50x coverage):
#   ./prepare_zymo_subset.sh \
#       zymo_d6300_raw.fastq.gz \
#       data/reference/ecoli_zymo_benchmark_strain.fasta \
#       benchmark_data/ecoli_subset \
#       50 \
#       4.8m

set -euo pipefail

RAW_READS="${1:?Usage: $0 <raw_reads.fastq.gz> <target_reference.fasta> <output_dir> [coverage] [genome_size]}"
TARGET_REF="${2:?Provide the target species reference FASTA}"
OUTPUT_DIR="${3:?Provide an output directory}"
TARGET_COVERAGE="${4:-50}"
GENOME_SIZE="${5:-4.8m}"

for tool in minimap2 samtools rasusa; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' not found on PATH. See docs/BENCHMARKING.md."; exit 1; }
done

mkdir -p "$OUTPUT_DIR"

echo "[1/4] Mapping raw Zymo reads to target reference ($TARGET_REF) ..."
minimap2 -ax map-ont -t "$(nproc)" "$TARGET_REF" "$RAW_READS" \
    | samtools sort -@ "$(nproc)" -o "$OUTPUT_DIR/mapped.bam" -
samtools index "$OUTPUT_DIR/mapped.bam"

echo "[2/4] Extracting mapped (target-species) reads only ..."
samtools fastq -F 0x904 "$OUTPUT_DIR/mapped.bam" > "$OUTPUT_DIR/target_species_reads.fastq"
# -F 0x904 excludes unmapped, secondary, and supplementary alignments --
# keeps one primary record per read that genuinely mapped to this species.

RAW_COUNT=$(( $(wc -l < "$OUTPUT_DIR/target_species_reads.fastq") / 4 ))
echo "    Extracted $RAW_COUNT reads before downsampling."

echo "[3/4] Downsampling to ${TARGET_COVERAGE}x coverage (genome size: $GENOME_SIZE) ..."
rasusa reads \
    --coverage "$TARGET_COVERAGE" \
    --genome-size "$GENOME_SIZE" \
    -o "$OUTPUT_DIR/subset_${TARGET_COVERAGE}x.fastq" \
    "$OUTPUT_DIR/target_species_reads.fastq"

FINAL_COUNT=$(( $(wc -l < "$OUTPUT_DIR/subset_${TARGET_COVERAGE}x.fastq") / 4 ))

echo "[4/4] Done."
echo "    Final subset: $OUTPUT_DIR/subset_${TARGET_COVERAGE}x.fastq ($FINAL_COUNT reads, target ${TARGET_COVERAGE}x)"
echo ""
echo "This subset is what benchmarking/run_comparison.sh expects as its READS input."
