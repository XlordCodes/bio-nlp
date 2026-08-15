"""
config.py
----------
Global, unified configuration. This file has ZERO internal dependencies
(per the project manifest) and is imported by virtually every other file.

Anything that needs to be identical across model/, data/, and backend/ --
vocabulary size, special token ids, the RLE base alphabet -- lives here,
and ONLY here. Before this file existed, hybrid_encoder.py and
rle_decoder.py each defined their own copies of these constants locally.
That's exactly the kind of drift risk (two files disagreeing on what
PAD_IDX is, say) the project manifest was written to prevent, so those
two files have been updated to import from here instead of redefining
these values themselves.
"""

# ---------------------------------------------------------------------------
# K-mer tokenization vocabulary
# ---------------------------------------------------------------------------
K = 6                      # hexamer window size
NUM_KMERS = 4 ** K         # 4096 unique A/C/G/T k-mers

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
NUM_SPECIAL_TOKENS = 4

VOCAB_SIZE = NUM_KMERS + NUM_SPECIAL_TOKENS  # 4100; k-mer ids are offset by NUM_SPECIAL_TOKENS

# ---------------------------------------------------------------------------
# Raw nucleotide alphabet (used by the tokenizer for k-mer <-> id conversion)
# ---------------------------------------------------------------------------
NUCLEOTIDES = ("A", "C", "G", "T")  # fixed order defines k-mer <-> integer mapping
VALID_INPUT_CHARS = frozenset("ATCGN")  # what raw sequence strings are allowed to contain

# ---------------------------------------------------------------------------
# RLE (Run-Length-Encoded homopolymer) auxiliary channel vocabulary
# ---------------------------------------------------------------------------
# Separate, small vocabulary for the raw nucleotide identity at each source
# position -- single-base resolution, NOT the k-mer vocabulary above.
RLE_BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
RLE_PAD_BASE_IDX = 5
RLE_BASE_VOCAB_SIZE = 6  # A, C, G, T, N, PAD

RLE_DEFAULT_MAX_RUN_LENGTH = 60  # homopolymer runs longer than this are clamped, not dropped

# ---------------------------------------------------------------------------
# Center-Base Mapping geometry (locked-in design decision)
# ---------------------------------------------------------------------------
# For k=6, the window has no single center (even length). The anchor is
# defined as the base at 0-indexed offset 2 within the window -- i.e. the
# 3rd nucleotide of the window counting from 1 ("index 3" in 1-indexed
# terms). This is the only anchor offset consistent with symmetric-ish
# padding of floor((k-1)/2) = 2 on the 5' end: it is what makes hexamer
# window 0's anchor land exactly on original base 0.
#
# Because k-1 = 5 is odd, the total padding needed to keep the number of
# hexamer tokens exactly equal to the number of original bases (N tokens
# for N bases) cannot be split evenly. The split is DELIBERATELY asymmetric:
#   left padding  = floor((k-1)/2) = 2
#   right padding = ceil((k-1)/2)  = 3
# See data/tokenizer.py for the derivation and a sanity check that proves
# this recovers every original base exactly.
RLE_ANCHOR_OFFSET = 2                        # 0-indexed position within each k-mer window
LEFT_PAD_BASES = (K - 1) // 2                # 2
RIGHT_PAD_BASES = (K - 1) - LEFT_PAD_BASES   # 3
PAD_NUCLEOTIDE = "N"           # fallback character; NOT used for boundary padding (see below), kept for
                                # anywhere a literal "unknown base" placeholder is genuinely needed.

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DEFAULT_REFERENCE_FASTA_PATH = "data/reference/ecoli_k12_mg1655.fasta"  # E. coli K-12 MG1655, NC_000913.3
# Used for SYNTHETIC TRAINING DATA generation (data/simulator.py + Badread) only.
# K-12 MG1655 is a standard, well-characterized lab reference -- fine as a
# simulation substrate regardless of strain, since Badread generates its own
# noisy reads FROM this sequence rather than using independently-sequenced
# real reads. Do NOT use this reference to benchmark against real Zymo
# sequencing data (see below).

BENCHMARK_REFERENCE_FASTA_PATH = "data/reference/ecoli_zymo_benchmark_strain.fasta"
# CONFIRMED via research (2026): this matches Zymo D6300's E. coli component,
# strain NRRL B-1109 (ST10 lineage, not a K-12 derivative). Definitive RefSeq
# assembly accession: GCF_005153645.1 (GenBank GCA_005153645.1); chromosome
# accession CP039753.1, reported length 4,773,399 bp + 1 plasmid.
#
# NOTE: our uploaded assembly's chromosome measured 4,804,267 bp via this
# project's own parsing -- ~31kb (~0.6%) longer than the 4,773,399 bp
# reported for CP039753.1. Plausibly just a different assembly
# version/submission of the same strain (contig trimming, gap-filling
# differences are common between assembly releases), but NOT byte-for-byte
# verified against CP039753.1 directly -- if exact reproducibility matters
# for a report, download CP039753.1 directly from NCBI rather than relying
# on this uploaded file's provenance.
#
# Used ONLY when benchmarking against real Zymo-derived reads -- see
# DEFAULT_REFERENCE_FASTA_PATH above for the training-data reference, and
# docs/BENCHMARKING.md for why these two must not be swapped.
#
# Contains 2 records (chromosome + plasmid) -- unlike DEFAULT_REFERENCE_FASTA_PATH,
# code that assumes single-record FASTA (e.g. data/simulator.py's load_reference)
# will reject this file as-is; the plasmid record needs separate handling if
# used for simulation rather than pure benchmarking alignment.

DEFAULT_TRAINING_DATA_JSONL_PATH = "data/training_pairs.jsonl"  # output of data/simulator.py, input to data/dataset.py

# Confirmed Zymo D6300 accessions (research, 2026) -- see BENCHMARK_REFERENCE_FASTA_PATH above.
ZYMO_ECOLI_REFSEQ_ACCESSION = "GCF_005153645.1"  # E. coli NRRL B-1109 (GenBank GCA_005153645.1)
ZYMO_ECOLI_CHROMOSOME_ACCESSION = "CP039753.1"   # reported length 4,773,399 bp
ZYMO_D6300_R10_ENA_RUN_ACCESSION = "ERR7287988"  # R10.4.1 chemistry (D6322, near-identical to D6300)
ZYMO_D6300_R9_ENA_RUN_ACCESSION = "ERR2906227"   # R9.4.1 chemistry -- older, prefer the R10 run above

# ---------------------------------------------------------------------------
# Backend / inference
# ---------------------------------------------------------------------------
MAX_INFERENCE_SEQUENCE_LENGTH = 200_000  # bp; safeguard against unbounded memory use from huge uploads
DEFAULT_INFERENCE_CHUNK_SIZE = 1024      # matches the model's stated max receptive field (Part 4 spec)
DEFAULT_INFERENCE_CHUNK_OVERLAP = 256    # locked-in design: 256-token overlap for 1024-token chunks
DEFAULT_DECODE_LENGTH_MARGIN = 64        # extra decode budget beyond chunk length, to allow for net insertions
DEFAULT_MODEL_CHECKPOINT_PATH = "checkpoints/model_best.pt"  # written by training/train.py, read by backend/main.py

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_EPOCHS = 5
DEFAULT_VAL_FRACTION = 0.1              # held out at the PAIR level, before chunking -- see train.py
DEFAULT_GRAD_CLIP_NORM = 5.0            # standard safeguard against RNN/LSTM exploding gradients

# Teacher forcing schedule. CHANGED after the first real 5-epoch run: decaying
# to 0.3 starting from step 0 caused train_loss AND val_loss to rise together
# from epoch 3 onward (val_loss 0.302 -> 0.454 by epoch 5) -- exposure-bias
# instability, not overfitting (train_loss rose too, not just val_loss).
#   1. END raised 0.3 -> 0.6: less reliance on the model's own rough
#      predictions as training input.
#   2. WARMUP_FRACTION delays decay start -- the first ~epoch now trains at
#      full teacher forcing before any decay begins, instead of decaying
#      from step 0.
DEFAULT_TEACHER_FORCING_START = 1.0
DEFAULT_TEACHER_FORCING_END = 0.6
DEFAULT_TEACHER_FORCING_WARMUP_FRACTION = 0.15

# LR scheduler. ADDED for the same reason: a constant LR meant nothing
# pulled training back once loss started rising -- Adam kept taking
# full-size steps into a worsening loss landscape for 3 straight epochs.
# patience=1 is deliberately short given only 5 epochs total.
DEFAULT_LR_SCHEDULER_FACTOR = 0.5
DEFAULT_LR_SCHEDULER_PATIENCE = 1
DEFAULT_LR_SCHEDULER_MIN_LR = 1e-5

# ---------------------------------------------------------------------------
# Sanity check: this file has no dependencies, so it should always be
# importable and internally consistent on its own.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    assert VOCAB_SIZE == 4100
    assert LEFT_PAD_BASES == 2
    assert RIGHT_PAD_BASES == 3
    assert LEFT_PAD_BASES + RIGHT_PAD_BASES == K - 1
    assert RLE_ANCHOR_OFFSET == LEFT_PAD_BASES, (
        "RLE_ANCHOR_OFFSET must equal LEFT_PAD_BASES for window 0's anchor "
        "to land on original base 0 -- see derivation above."
    )
    print("config.py self-check passed.")
    print(f"VOCAB_SIZE={VOCAB_SIZE}, LEFT_PAD_BASES={LEFT_PAD_BASES}, "
          f"RIGHT_PAD_BASES={RIGHT_PAD_BASES}, RLE_ANCHOR_OFFSET={RLE_ANCHOR_OFFSET}")
