"""
backend/schemas.py
--------------------
Pydantic (v2) request/response contracts for the inference API. Pure data
contracts and validation only -- no model loading, no tokenization, no
FASTA-file parsing (multi-record file parsing belongs to
backend/inference_engine.py, which is what actually touches the model).

Validation performed here catches malformed input BEFORE it ever reaches
the (expensive) model forward pass: wrong alphabet, empty string, or a
sequence long enough to threaten memory get rejected at the HTTP layer with
a clear 422 error, not a confusing failure three layers down inside the
encoder.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import VALID_INPUT_CHARS, MAX_INFERENCE_SEQUENCE_LENGTH


class InferenceRequest(BaseModel):
    """
    Request body for raw-string sequence submission (POST /correct).
    Multipart .fasta/.fa FILE uploads are handled as a separate FastAPI
    endpoint parameter (UploadFile), not through this schema -- see
    backend/main.py.
    """

    model_config = ConfigDict(str_strip_whitespace=False)  # we do our own, more thorough normalization below

    sequence: str = Field(
        ...,
        min_length=1,
        description="Raw nucleotide sequence (A/C/G/T/N). A single leading FASTA header line "
                    "('>...') is tolerated and stripped; internal whitespace/newlines are removed.",
    )

    @field_validator("sequence")
    @classmethod
    def normalize_and_validate_sequence(cls, value: str) -> str:
        # Tolerate a pasted FASTA block: drop any header line(s), join the rest.
        lines = [line for line in value.splitlines() if not line.strip().startswith(">")]
        joined = "".join(line.strip() for line in lines)
        cleaned = joined.upper()

        if len(cleaned) == 0:
            raise ValueError(
                "Sequence is empty after stripping FASTA header lines and whitespace. "
                "Provide at least one nucleotide."
            )

        if len(cleaned) > MAX_INFERENCE_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence length ({len(cleaned):,} bp) exceeds the maximum allowed "
                f"({MAX_INFERENCE_SEQUENCE_LENGTH:,} bp). Submit a shorter region, or split "
                f"the read/genome into smaller pieces before uploading."
            )

        invalid_chars = set(cleaned) - VALID_INPUT_CHARS
        if invalid_chars:
            raise ValueError(
                f"Sequence contains invalid character(s) {sorted(invalid_chars)}; only "
                f"{sorted(VALID_INPUT_CHARS)} are permitted."
            )

        return cleaned


class CorrectionMetrics(BaseModel):
    """
    Statistics describing how the corrected output differs from the
    original INPUT sequence (not from any hidden ground truth -- at real
    inference time on an unknown read, there is no ground truth to compare
    against; edit distance here quantifies "how much the model changed",
    not "how correct the change was". Ground-truth-based accuracy metrics
    belong to training/evaluate.py, where the true clean sequence is known.
    """

    input_length: int = Field(..., description="Length of the (cleaned) input sequence, in bases.")
    corrected_length: int = Field(..., description="Length of the corrected output sequence, in bases.")
    edit_distance: int = Field(
        ..., description="Levenshtein edit distance (via edlib) between input and corrected_sequence."
    )
    num_matches: int = Field(..., description="Aligned positions where input and corrected agree.")
    num_substitutions: int = Field(..., description="Aligned positions where input and corrected disagree (same length, different base).")
    num_insertions: int = Field(..., description="Bases present in corrected_sequence with no counterpart in input.")
    num_deletions: int = Field(..., description="Bases present in input with no counterpart in corrected_sequence.")
    num_chunks: int = Field(..., description="Number of model inference chunks the input was split into.")
    latency_ms: float = Field(..., description="Wall-clock inference time in milliseconds.")
    chunks_without_eos: int = Field(
    0, description="Chunks in this sequence whose decoder never emitted <EOS> "
                   "within its decode budget; output for those chunks may be truncated.")   

class AttentionChunk(BaseModel):
    """
    One chunk's worth of decoder cross-attention, for the frontend XAI
    heatmap. A list rather than one monolithic matrix, because long
    sequences are split into overlapping chunks (see inference_engine.py's
    stitching logic) -- each chunk's decoder attended only over its own
    local receptive field, so a single combined matrix across the whole
    corrected sequence isn't well-defined. corrected_start/corrected_end
    give the frontend the coordinates needed to place each chunk's heatmap
    at the right position along the final corrected_sequence.
    """

    chunk_index: int
    corrected_start: int = Field(..., description="Start offset (inclusive) of this chunk's contribution in corrected_sequence.")
    corrected_end: int = Field(..., description="End offset (exclusive) of this chunk's contribution in corrected_sequence.")
    source_start: int = Field(..., description="Start offset (inclusive) of this chunk in the original input sequence.")
    source_end: int = Field(..., description="End offset (exclusive) of this chunk in the original input sequence.")
    attention_matrix: List[List[float]] = Field(
        ..., description="Shape (decode_steps, source_chunk_length); attention_matrix[t][i] = alpha_(t,i)."
    )


class InferenceResponse(BaseModel):
    corrected_sequence: str
    metrics: CorrectionMetrics
    attention_chunks: List[AttentionChunk]
    per_base_confidence: List[float] = Field(
        ...,
        description=(
            "Per-base confidence for corrected_sequence, same length and index "
            "alignment as corrected_sequence. Each value is the softmax "
            "probability the model assigned to the base it actually chose at "
            "that position (0-1). Low values flag positions worth human or "
            "wet-lab review -- this is NOT a guarantee a correction is right, "
            "and does not by itself distinguish a genuine rare biological "
            "variant from a sequencing error; see docs for the reasoning."
        ),
    )


class ErrorResponse(BaseModel):
    detail: str


class HealthCheckResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    device: str


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pydantic import ValidationError

    # -- 1. Valid plain sequence passes and is uppercased -----------------
    req = InferenceRequest(sequence="acgtACGTnN")
    assert req.sequence == "ACGTACGTNN", req.sequence
    print("[1/5] Plain-sequence normalization (case + passthrough) passed.")

    # -- 2. FASTA header line is stripped, whitespace/newlines removed -----
    fasta_block = ">some header, not real data\nACGT\nACGT\n  ACGT  \n"
    req = InferenceRequest(sequence=fasta_block)
    assert req.sequence == "ACGTACGTACGT", req.sequence
    print("[2/5] FASTA header stripping + whitespace normalization passed.")

    # -- 3. Empty / whitespace-only sequence is rejected --------------------
    try:
        InferenceRequest(sequence="   \n  ")
        raise AssertionError("Expected ValidationError for empty sequence")
    except ValidationError:
        pass
    print("[3/5] Empty-sequence rejection passed.")

    # -- 4. Invalid alphabet is rejected -------------------------------------
    try:
        InferenceRequest(sequence="ACGTXYZ")
        raise AssertionError("Expected ValidationError for invalid characters")
    except ValidationError as e:
        assert "invalid character" in str(e)
    print("[4/5] Invalid-alphabet rejection passed.")

    # -- 5. Oversized sequence is rejected ------------------------------------
    too_long = "A" * (MAX_INFERENCE_SEQUENCE_LENGTH + 1)
    try:
        InferenceRequest(sequence=too_long)
        raise AssertionError("Expected ValidationError for oversized sequence")
    except ValidationError as e:
        assert "exceeds the maximum" in str(e)
    print("[5/5] Oversized-sequence rejection passed.")

    # -- Response schema construction + JSON round-trip sanity ---------------
    response = InferenceResponse(
        corrected_sequence="ACGTACGT",
        metrics=CorrectionMetrics(
            input_length=8, corrected_length=8, edit_distance=1, num_matches=7,
            num_substitutions=1, num_insertions=0, num_deletions=0, num_chunks=1, latency_ms=12.3,
        ),
        attention_chunks=[
            AttentionChunk(
                chunk_index=0, corrected_start=0, corrected_end=8, source_start=0, source_end=8,
                attention_matrix=[[0.1, 0.9], [0.5, 0.5]],
            )
        ],
        per_base_confidence=[0.99, 0.98, 0.95, 0.91, 0.97, 0.99, 0.92, 0.96],
    )
    round_tripped = InferenceResponse.model_validate_json(response.model_dump_json())
    assert round_tripped == response
    print("Response schema construction + JSON round-trip passed.")

    print("\nAll schema sanity checks passed.")