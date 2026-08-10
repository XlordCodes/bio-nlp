# Benchmarking Runbook

Everything in this document runs on **your own machine**, not in the sandbox this
project was developed in. The sandbox has no GPU and is not where real training or
tool comparisons should happen -- it's where the code was written and syntax-checked.

## 0. Why there are two E. coli references

- `data/reference/ecoli_k12_mg1655.fasta` -- K-12 MG1655. Used only as the substrate
  for **synthetic training data** (Badread simulates noisy reads *from* this sequence).
  Fine for training regardless of strain, since Badread generates its own reads rather
  than using independently-sequenced real data.
- `data/reference/ecoli_zymo_benchmark_strain.fasta` -- E. coli strain **NRRL B-1109**
  (Zymo D6300's E. coli component, ST10 lineage, not a K-12 derivative). **Strain
  identity CONFIRMED** via research: RefSeq assembly `GCF_005153645.1` (GenBank
  `GCA_005153645.1`), chromosome accession `CP039753.1`, reported length 4,773,399 bp.
  Our uploaded assembly's chromosome measured 4,804,267 bp -- ~31kb (~0.6%) longer,
  plausibly a different assembly version/submission of the same strain, but not
  byte-for-byte verified against `CP039753.1` directly. If exact reproducibility
  matters for a report, download `CP039753.1` from NCBI directly rather than relying
  on this upload's provenance.

**Rule going forward: use K-12 MG1655 for training data generation. Use the Zymo
benchmark strain reference ONLY when aligning/scoring against real Zymo-derived
reads.** Using the wrong one for benchmarking will silently inflate error metrics
with real strain-level SNPs, not sequencing errors.

## 1. Installing the comparison tools

```bash
# Racon -- CPU-native, build from source
git clone https://github.com/lbcb-sci/racon.git --recursive
cd racon && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
# racon binary is now in build/bin/racon -- add to PATH

# Medaka -- CPU-only install (avoids pulling unnecessary CUDA binaries)
python3 -m venv medaka-env && source medaka-env/bin/activate
pip install medaka --extra-index-url https://download.pytorch.org/whl/cpu
# Also requires on PATH: samtools >= 1.14, minimap2 >= 2.17, tabix, bgzip
# CPU execution works but is slow (minutes to hours depending on pileup depth) --
# this is expected, not a misconfiguration.

# pomoxis -- ONT's own assessment suite (assess_assembly, assess_homopolymers)
# v0.4.2 (July 2026) confirmed current.
pip install pomoxis
# or: conda install -c bioconda pomoxis

# dnadiff (MUMmer4) -- SNP/indel counting. conda is simpler than building from
# source (source-build instructions kept below in case conda isn't available).
conda install -c bioconda mummer4
# or from source:
#   git clone https://github.com/mummer4/mummer.git
#   cd mummer && autoreconf -fi && ./configure && make -j$(nproc) && sudo make install

# Rasusa -- coverage-normalized downsampling. v0.4.0+ confirmed current.
cargo binstall rasusa   # installs a prebuilt binary, faster than `cargo install`
# or: conda install -c bioconda rasusa

# Flye -- assembler used in all three pipeline comparisons
pip install flye
```

**dorado correct (HERRO) is deliberately not in the automated comparison script.**
Per research it needs 64+ CPU cores, 256GB+ RAM, and 32GB+ VRAM to run practically --
data-center hardware, not a workstation. Confirmed distribution: precompiled static
binaries via Oxford Nanopore's GitHub releases (Linux x86_64/ARM64, macOS, Windows);
it auto-downloads HERRO model weights at runtime (offline download also supported).
It supports split-processing -- the CPU-bound mapping stage (`--to-paf`) can run
separately from the GPU-bound inference stage (`--from-paf`), useful if you only have
GPU access intermittently. If you have access to suitable hardware, install and run
manually:

```bash
dorado correct reads.fastq.gz --index-size 4G --batch-size 32 > corrected.fasta
# --index-size and --batch-size are throttled down from ONT's whole-genome defaults
# specifically to fit a single bacterial genome on consumer-class hardware.
```

**Worth knowing for positioning, not just caution:** research reports HERRO's full
pipeline (all-vs-all minimap2 pileups + Transformer inference) taking 100+ wall-clock
*hours* for 50x human genomes even on multi-GPU A100/L40 clusters -- but estimates a
5Mb bacterial genome on a single RTX 4090 in *minutes* for a single-read approach like
ours, specifically because we skip HERRO's expensive all-vs-all overlap computation
entirely. Our per-token autoregressive decode may be slower than HERRO's parallel
Transformer, but avoiding MSA construction altogether may still make us faster
end-to-end at bacterial scale -- worth measuring directly, not assuming either way.

## 2. Preparing benchmark data

If you have real Zymo D6300 reads, **prefer the R10.4.1 chemistry run**
(`ERR7287988`, D6322 -- near-identical to D6300) over the older R9.4.1 run
(`ERR2906227`), since this project targets modern chemistry throughout. Both
hosted on ENA under project `PRJEB29504`. Extract a tractable single-species
subset against the confirmed reference (`GCF_005153645.1` / `CP039753.1`):

```bash
./benchmarking/prepare_zymo_subset.sh \
    zymo_d6300_r10_raw.fastq.gz \
    data/reference/ecoli_zymo_benchmark_strain.fasta \
    benchmark_data/ecoli_subset \
    50 \
    4.8m
```

This maps raw reads to the target reference, extracts only reads that genuinely
mapped to E. coli (discarding the other 9 organisms in the mock community), and
downsamples to 50x coverage -- standard practice for a compute-constrained
proof-of-concept, and still directly citable against full-community benchmarks.

**If using the D6331 Gut standard instead of D6300**, note it contains a *blend of
five different E. coli strains* (including B-1109, JM109, B-3008), which requires
phased evaluation to separate correctly -- D6300's single E. coli strain is
substantially simpler and is what this project's scripts assume.

If instead you're just testing against synthetic data (no real Zymo reads yet),
generate reads the way this project already does:

```bash
python -m data.simulator \
    --reference data/reference/ecoli_k12_mg1655.fasta \
    --output data/training_pairs.jsonl \
    --quantity 50x
```

## 3. Training for real

The smoke-test scale used during development (~100bp truncated pairs, a handful of
epochs) proves the training loop works -- it is **not** a real training run. For an
actual result:

```bash
python -m training.train \
    --train-data data/training_pairs.jsonl \
    --epochs 50 \
    --batch-size 16 \
    --chunk-size 1024
```

Expect this to take substantially longer than anything run in the sandbox --
real convergence on genome-scale data, especially with the autoregressive decoder,
is measured in hours on a GPU, likely much longer on CPU. This is exactly the
compute-vs-architecture tradeoff flagged in the roadmap (Stage 2/3).

## 4. Running the full comparison

Once you have a real checkpoint and a prepared read set:

```bash
./benchmarking/run_comparison.sh \
    benchmark_data/ecoli_subset/subset_50x.fastq \
    data/reference/ecoli_zymo_benchmark_strain.fasta \
    checkpoints/model_best.pt \
    benchmark_data/results
```

This runs three pipelines against the same reads and reference:
- **A -- Uncorrected**: Flye alone, no polishing.
- **B -- Standard industry pipeline**: Flye → Racon (×4) → Medaka.
- **C -- Ours**: our model corrects reads first, then Flye, with **no** subsequent
  Racon/Medaka polishing.

Then assesses all three with `pomoxis assess_assembly` (Q-scores, indel/mismatch
breakdown), `pomoxis assess_homopolymers` (homopolymer-length-stratified accuracy --
**this is the specific number that validates or invalidates the RLE-channel design
decision**), and `dnadiff` (exact SNP/indel counts).

### What "success" looks like, concretely

From the research's published baselines: Racon-alone leaves ~9,500 residual errors
on a typical bacterial genome; Racon→Medaka brings that to ~1,900-3,000, with an
indel:mismatch ratio around **39:1-40:1** and **86%+ of errors in homopolymer
regions**. Our target:

- Beat the uncorrected baseline by a wide margin (low bar, sanity check).
- Get meaningfully below the 39:1-40:1 indel:mismatch ratio, driven specifically by
  homopolymer-region improvement -- this is the number that proves the RLE channel
  earns its complexity, not just that the model works at all.
- If pipeline C's assembly needs *no* further Racon/Medaka polishing to match
  pipeline B's final quality, that's the strongest possible result: pre-assembly
  single-read correction replacing the standard two-stage polish entirely.

## 5. Testing for reference hallucination (variant preservation)

This targets the specific failure mode research flagged for single-read (non-MSA)
correctors -- see `training/variant_preservation_test.py`'s module docstring for
full methodology.

```bash
python -m training.variant_preservation_test \
    --checkpoint checkpoints/model_best.pt \
    --reference data/reference/ecoli_k12_mg1655.fasta \
    --num-chunks 50 \
    --mutations-per-chunk 3
```

Reports a `variant_preservation_rate` -- fraction of genuine planted point mutations
the model correctly preserved vs. reverted back to the reference it trained on.
Closer to 1.0 is better. This is a post-training validation step; on an untrained
model every outcome is noise.

## 6. Reporting throughput honestly

Per research, always report Mb/s **alongside the exact hardware used** -- CPU
model, or GPU model + VRAM, or it isn't comparable to anything. `correct_fastq.py`
already reports this automatically. Compare against research's cited figures:
Medaka completes a bacterial polish in minutes on an A100/H100, hours on CPU;
dorado correct effectively requires data-center GPUs to be practical at all.

## 7. Known gaps, stated plainly

- **No methylation modeling.** Badread-simulated training data doesn't encode
  Dam/Dcm-correlated substitution bias (A↔G transitions at GATC/CCWGG motifs) that
  real R10.4.1 data has. This is a real, currently-unaddressed gap, not an oversight
  to paper over in any report or presentation of results.
- **The Zymo benchmark reference is strain-confirmed (NRRL B-1109, `GCF_005153645.1`)
  but not byte-exact verified** against the canonical `CP039753.1` chromosome record
  (~0.6% length difference, likely just assembly-version drift). Fine for
  benchmarking; download `CP039753.1` directly if exact reproducibility matters.
