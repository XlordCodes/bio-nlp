# Context-Driven Neural Sequence Translation for Prokaryotic Genome Repair

A Seq2Seq neural framework for correcting Oxford Nanopore (ONT) long-read
sequencing errors -- insertions, deletions, and homopolymer artifacts --
without relying on secondary short-read data. DNA is tokenized as
overlapping hexamers and translated by a hybrid CNN + BiLSTM encoder with
cross-attention, an auxiliary run-length-encoded (RLE) channel for
homopolymer tracking, and an autoregressive decoder whose attention weights
are exposed for explainability.

**Current status: architecture, pipeline, and tooling are complete and
tested end-to-end; no full training run has been performed yet.** The
backend will serve an untrained (random-weight) model, with a loud startup
warning, until `training/train.py` is run against a real dataset and
produces a checkpoint at `checkpoints/model_best.pt`. Every module below has
been verified to run correctly against real data (see each file's own
`__main__` block and `tests/`), but no accuracy/identity numbers exist yet
for this system as a whole -- don't quote any until `training/evaluate.py`
has actually been run against a trained checkpoint.

## Project layout

```
project-root/
├── config.py                      # single source of truth for shared constants
├── requirements.txt
│
├── model/                         # PyTorch architecture
│   ├── hybrid_encoder.py          #   1D-CNN + BiLSTM + fusion
│   ├── cross_attention.py         #   Bahdanau attention, returns alpha
│   ├── rle_decoder.py             #   autoregressive decoder + RLE fusion
│   └── sequence_translation_model.py  # ties encoder+decoder together
│
├── data/                          # data pipeline
│   ├── reference/ecoli_k12_mg1655.fasta   # E. coli K-12 MG1655, NC_000913.3
│   ├── simulator.py                # Badread + minimap2 -> aligned training pairs
│   ├── tokenizer.py                 # k-mer + RLE tokenization
│   └── dataset.py                   # alignment-aware chunking, batching
│
├── training/
│   ├── train.py                     # training loop, checkpointing
│   ├── evaluate.py                  # full biological validation + FASTA dump
│   └── metrics.py                   # edit distance, frame preservation, minimap2
│
├── backend/                        # FastAPI serving layer
│   ├── schemas.py
│   ├── inference_engine.py          # chunking, stitching, metrics
│   └── main.py
│
├── frontend/                       # React + Tailwind dashboard
│   └── src/...
│
└── tests/                          # fast unit tests (pytest)
```


## Setup

### 1. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers everything except **minimap2**, which is a system
binary, not a Python package:

```bash
apt-get install minimap2       # Debian/Ubuntu
# or: conda install -c bioconda minimap2
```

### 2. Reference genome

The project ships with `data/reference/ecoli_k12_mg1655.fasta` (E. coli
K-12 MG1655, RefSeq `NC_000913.3`, downloaded from NCBI). If you need to
re-download it: search `NC_000913.3` at
[ncbi.nlm.nih.gov/nuccore](https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3) →
Send to → File → Format: FASTA.

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env.production   # only needed for a production build; edit the API URL
```

## Running things

**Generate training data** (real Badread simulation + real minimap2
alignment against the reference genome):
```bash
python -m data.simulator --reference data/reference/ecoli_k12_mg1655.fasta \
    --output data/training_pairs.jsonl --quantity 50x
```

**Train**:
```bash
python -m training.train --train-data data/training_pairs.jsonl --epochs 20
```
Writes the best checkpoint to `checkpoints/model_best.pt` (via
`backend.inference_engine.save_checkpoint` -- this is the only code path
that should ever write a checkpoint, so the format always matches what
`backend/main.py` expects to load).

**Evaluate** a trained checkpoint against held-out ground truth (dumps
FASTA + a JSON summary):
```bash
python -m training.evaluate --checkpoint checkpoints/model_best.pt \
    --validation-data data/validation_pairs.jsonl --output-dir evaluation_output
```

**Run the backend**:
```bash
uvicorn backend.main:app --reload
```

**Run the frontend** (in a second terminal):
```bash
cd frontend && npm run dev
```

**Run the fast unit tests**:
```bash
pytest tests/
```

**Run any module's own integration self-check** (real data, no mocking --
slower than `tests/`, see `tests/README.md` for the distinction):
```bash
python -m data.tokenizer
python -m data.simulator
python -m backend.inference_engine
# ...etc -- every module under model/, data/, backend/, training/ has one
```

## Design notes worth knowing before modifying anything

A few non-obvious decisions are documented in detail in the files
themselves (each has extensive module/function docstrings explaining the
*why*, not just the *what*) -- worth reading before changing:

- **`config.py`** is the single source of truth for shared constants
  (vocab size, special token ids, padding geometry). Model and data files
  import from it rather than redefining constants locally, specifically to
  prevent silent drift between modules.
- **`data/tokenizer.py`** pads sequence boundaries by edge-replicating the
  sequence's own first/last base, not with a literal placeholder character
  -- the latter was tried, caught by testing, and found to force every
  sequence's boundary tokens to `<UNK>` regardless of real ambiguity.
- **`data/dataset.py`** chunks long reads using real alignment breakpoints,
  not fixed character offsets -- noisy and clean sequences drift out of
  index-alignment after any indel, so naive chunking would silently pair
  mismatched regions.
- **`backend/inference_engine.py`** stitches overlapping chunk outputs back
  together via edlib alignment at each chunk boundary, splitting at the
  alignment midpoint so each chunk contributes the half of its overlap
  closer to its own decoding center.
- Anywhere `edlib` is used, check the comment on its query/target CIGAR
  convention before trusting `I`/`D` labels at a glance -- this caused a
  real bug during development (see `tests/test_inference_engine.py` and
  `tests/test_metrics.py` for the regression tests that now guard it).
