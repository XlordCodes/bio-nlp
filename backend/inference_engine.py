"""
backend/inference_engine.py
------------------------------
The actual inference pipeline: raw string -> tokenize -> (chunk if needed)
-> model forward pass under torch.no_grad() -> decode -> stitch chunks back
together -> metrics -> a plain dict matching schemas.InferenceResponse's
shape exactly (backend/main.py just wraps this dict in the Pydantic model).

[Raw String] -> [Hexamer Tokenizer] -> [Tensor Conversion]
                                              |
[JSON Response] <- [Decode + Stitch] <- [Model Forward Pass, torch.no_grad()]

-----------------------------------------------------------------------------
THE STITCHING PROBLEM (why this isn't just string concatenation)
-----------------------------------------------------------------------------
Long sequences are split into overlapping chunks (locked design: 1024-token
chunks, 256-token overlap) so the decoder's receptive field is never
overwhelmed. But each chunk is corrected by an INDEPENDENT decoder run, and
because indel correction can change sequence length, chunk i's corrected
output and chunk i+1's corrected output will not, in general, be the exact
same length in their shared overlap region -- so naive fixed-index
concatenation would either duplicate or drop bases at every seam, and would
compound across a long read with many chunks.

Fix: for every pair of adjacent chunks, take a window of characters from
chunk i's trailing edge ("tail") and chunk i+1's leading edge ("head") --
both meant to represent the same underlying genomic region -- and align
them against each other with edlib (global/"NW" alignment, i.e.
Needleman-Wunsch). The splice point is chosen at the MIDPOINT of that
alignment: chunk i contributes the half of the overlap closer to its own
center (where its decoder had the most surrounding context), and chunk i+1
contributes the half closer to ITS center. This is the "prioritize
center-mass predictions over edge predictions" design locked in during
architecture planning, made concrete.
"""

import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import edlib  # type: ignore
import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler  # type: ignore

from config import (
    DEFAULT_INFERENCE_CHUNK_SIZE,
    DEFAULT_INFERENCE_CHUNK_OVERLAP,
    DEFAULT_DECODE_LENGTH_MARGIN,
    EOS_IDX,
)
from data.tokenizer import KmerTokenizer
from model.sequence_translation_model import SequenceTranslationModel, SequenceTranslationConfig


class InferenceEngine:
    """
    Wraps a loaded SequenceTranslationModel with everything needed to serve
    real inference requests: chunking long inputs, running the model,
    stitching chunk outputs back together, and computing response metrics.
    Holds the model in eval() mode for its entire lifetime.
    """

    def __init__(
        self,
        model: SequenceTranslationModel,
        tokenizer: Optional[KmerTokenizer] = None,
        device: Optional[torch.device] = None,
        chunk_size: int = DEFAULT_INFERENCE_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_INFERENCE_CHUNK_OVERLAP,
        decode_length_margin: int = DEFAULT_DECODE_LENGTH_MARGIN,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})."
            )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer or KmerTokenizer()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.decode_length_margin = decode_length_margin
        
    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, device: Optional[torch.device] = None, **engine_kwargs
    ) -> "InferenceEngine":
        """
        CHECKPOINT FORMAT CONTRACT: expects the exact structure written by
        `save_checkpoint()` in this file --
            {"model_state_dict": <state_dict>, "model_config": SequenceTranslationConfig}
        training/train.py must call save_checkpoint() (not roll its own
        torch.save) so the writer and this reader can never drift apart.

        weights_only=False is used deliberately: this checkpoint format
        intentionally stores a SequenceTranslationConfig dataclass alongside
        the tensors, not just tensors, and this loader only ever reads
        checkpoints this project itself produced -- it is not loading
        arbitrary untrusted files.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} is missing 'model_state_dict' and/or "
                f"'model_config'. Checkpoints must be written via "
                f"backend.inference_engine.save_checkpoint() to match this loader's contract."
            )

        model = SequenceTranslationModel(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, device=device, **engine_kwargs)

    # -- chunking -----------------------------------------------------------

    def _chunk_offsets(self, sequence_length: int) -> List[Tuple[int, int]]:
        """Returns [(start, end), ...] offsets into the source sequence, each chunk overlapping the next by self.chunk_overlap bases."""
        if sequence_length <= self.chunk_size:
            return [(0, sequence_length)]
        stride = self.chunk_size - self.chunk_overlap
        offsets = []
        start = 0
        while True:
            end = min(start + self.chunk_size, sequence_length)
            offsets.append((start, end))
            if end == sequence_length:
                break
            start += stride
        return offsets

    # -- per-chunk inference --------------------------------------------------

    @torch.no_grad()
    def _correct_chunk(self, chunk_seq: str) -> Tuple[str, List[List[float]], List[float], bool]:
        tensors = self.tokenizer.encode_to_tensors(chunk_seq)
        src_tokens = tensors["token_ids"].unsqueeze(0).to(self.device)
        src_lengths = tensors["length"].unsqueeze(0).to(self.device)
        rle_base_ids = tensors["rle_base_ids"].unsqueeze(0).to(self.device)
        rle_run_lengths = tensors["rle_run_lengths"].unsqueeze(0).to(self.device)

        # Decoder gets a little extra budget beyond the input length, since a
        # net-insertion-heavy correction could make the clean sequence
        # slightly longer than the noisy input.
        max_decode_len = len(chunk_seq) + self.decode_length_margin

        output = self.model.predict(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            rle_base_ids=rle_base_ids,
            rle_run_lengths=rle_run_lengths,
            max_decode_len=max_decode_len,
        )

        predicted_tokens = output.predicted_tokens[0].tolist()
        eos_emitted = EOS_IDX in predicted_tokens
        # NOTE: no per-chunk print here -- on a real evaluation run this can
        # fire on a meaningful fraction of chunks (expected, especially on an
        # early-training checkpoint), and printing once per chunk floods the
        # log without being any more actionable. See correct_sequence()'s
        # aggregate NOTE line, and training/evaluate.py's summary, which
        # report the RATE across a run instead -- the number actually worth
        # watching.

        # Per-step confidence: the probability the model assigned to the
        # token it actually chose at each decode step (max softmax prob).
        # logits are already fully computed for every step regardless --
        # this reads data that exists anyway, no extra forward pass.
        step_probs = F.softmax(output.logits[0], dim=-1)  # (T', vocab_size)
        step_confidences = step_probs.max(dim=-1).values.detach().cpu().tolist()

        corrected_chunk, confidences = self.tokenizer.decode_with_confidence(
            predicted_tokens, step_confidences
        )
        attention_matrix = output.attention_matrix[0].detach().cpu().tolist()
        return corrected_chunk, attention_matrix, confidences, eos_emitted

    # -- stitching --------------------------------------------------------------

    def _find_splice_point(self, tail: str, head: str) -> Tuple[int, int]:
        """
        Aligns `tail` (chunk i's trailing edge) against `head` (chunk i+1's
        leading edge) and returns (tail_keep_len, head_skip_len) at the
        alignment's midpoint. See module docstring for the reasoning.
        """
        if len(tail) == 0 or len(head) == 0:
            return len(tail), 0

        result = edlib.align(tail, head, mode="NW", task="path")
        nice = edlib.getNiceAlignment(result, tail, head)
        aligned_len = len(nice["query_aligned"])
        mid_aligned_idx = aligned_len // 2

        tail_pos = 0
        head_pos = 0
        for i in range(mid_aligned_idx):
            if nice["query_aligned"][i] != "-":
                tail_pos += 1
            if nice["target_aligned"][i] != "-":
                head_pos += 1
        return tail_pos, head_pos

    def _stitch_chunks(
        self,
        chunk_offsets: List[Tuple[int, int]],
        corrected_chunks: List[str],
        confidence_chunks: Optional[List[List[float]]] = None,
    ):
        """
        Returns (final_corrected_sequence, corrected_offsets) where
        corrected_offsets[i] is the (start, end) slice of final_corrected_sequence
        that chunk i actually contributed -- needed so the API response can
        tell the frontend where each chunk's attention_matrix belongs.

        If confidence_chunks is given (confidence_chunks[i] is a list the
        same length as corrected_chunks[i], one value per base), also
        returns final_confidences as a third element, spliced with the exact
        same [start:end] indices as the sequence itself so per-base
        confidence can never drift out of alignment with the sequence it
        describes. confidence_chunks is optional and keyword-compatible with
        the original 2-argument/2-return-value form specifically so existing
        callers (including tests/test_inference_engine.py, which calls this
        directly) don't need to change at all.
        """
        n = len(corrected_chunks)
        if n == 1:
            if confidence_chunks is None:
                return corrected_chunks[0], [(0, len(corrected_chunks[0]))]
            return corrected_chunks[0], [(0, len(corrected_chunks[0]))], list(confidence_chunks[0])

        contrib_start = [0] * n
        contrib_end = [len(c) for c in corrected_chunks]

        for i in range(n - 1):
            noisy_start_i, noisy_end_i = chunk_offsets[i]
            noisy_start_next, _ = chunk_offsets[i + 1]
            overlap_len_noisy = max(noisy_end_i - noisy_start_next, 1)

            corrected_i = corrected_chunks[i]
            corrected_next = corrected_chunks[i + 1]

            tail_window = min(overlap_len_noisy, len(corrected_i))
            head_window = min(overlap_len_noisy, len(corrected_next))
            tail = corrected_i[-tail_window:] if tail_window > 0 else ""
            head = corrected_next[:head_window] if head_window > 0 else ""

            tail_keep, head_skip = self._find_splice_point(tail, head)

            contrib_end[i] = len(corrected_i) - (len(tail) - tail_keep)
            contrib_start[i + 1] = head_skip

        final_parts = []
        final_confidence_parts: Optional[List[float]] = [] if confidence_chunks is not None else None
        corrected_offsets = []
        cursor = 0
        for i in range(n):
            start, end = contrib_start[i], contrib_end[i]
            if start > end:
                # Defensive: two very short/divergent chunks could in principle
                # produce a crossed-over splice; clamp to an empty contribution
                # rather than let a negative-length slice silently misbehave.
                start = end
            piece = corrected_chunks[i][start:end]
            final_parts.append(piece)
            if confidence_chunks is not None and final_confidence_parts is not None:
                final_confidence_parts.extend(confidence_chunks[i][start:end])
            corrected_offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)

        if confidence_chunks is None:
            return "".join(final_parts), corrected_offsets
        return "".join(final_parts), corrected_offsets, final_confidence_parts
    
    # -- metrics ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        input_sequence: str, corrected_sequence: str, num_chunks: int, latency_ms: float
    ) -> dict:
        """
        Edit distance and edit-type breakdown between the ORIGINAL INPUT and
        the corrected output (not vs. any hidden ground truth -- see
        schemas.CorrectionMetrics docstring for why).

        edlib.align(query, target, ...) convention (confirmed empirically,
        not assumed -- see chat): cigar op 'I' marks a base present in the
        QUERY but absent from the TARGET; 'D' marks a base present in the
        TARGET but absent from the QUERY. Here query=input_sequence,
        target=corrected_sequence, so:
          - 'I' = a base that was in the input and is GONE from the output
                   -> the model DELETED it -> counts toward num_deletions.
          - 'D' = a base that is in the output but was NOT in the input
                   -> the model INSERTED it -> counts toward num_insertions.
        This is the opposite of what the op letters suggest at a glance,
        which is exactly why it's called out explicitly here rather than
        left to a reader's (or future editor's) intuition.
        """
        result = edlib.align(input_sequence, corrected_sequence, mode="NW", task="path")
        counts = {"=": 0, "X": 0, "I": 0, "D": 0}
        for length_str, op in re.findall(r"(\d+)([=XID])", result["cigar"]):
            counts[op] += int(length_str)

        return {
            "input_length": len(input_sequence),
            "corrected_length": len(corrected_sequence),
            "edit_distance": result["editDistance"],
            "num_matches": counts["="],
            "num_substitutions": counts["X"],
            "num_insertions": counts["D"],   # see docstring: target-only bases == insertions into the output
            "num_deletions": counts["I"],    # see docstring: query-only bases == deletions from the input
            "num_chunks": num_chunks,
            "latency_ms": latency_ms,
        }

    # -- public entry point -----------------------------------------------------

    def correct_sequence(self, raw_sequence: str) -> dict:
        """
        Full pipeline. Returns a plain dict matching schemas.InferenceResponse's
        field names exactly -- backend/main.py does `InferenceResponse(**result)`.
        """
        if len(raw_sequence) == 0:
            raise ValueError("Cannot correct an empty sequence.")

        start_time = time.perf_counter()
        sequence = raw_sequence.upper()

        offsets = self._chunk_offsets(len(sequence))

        corrected_chunks: List[str] = []
        attention_matrices: List[List[List[float]]] = []
        confidence_chunks: List[List[float]] = []
        chunks_without_eos = 0
        for chunk_start, chunk_end in offsets:
            corrected_chunk, attention_matrix, confidences, eos_emitted = self._correct_chunk(
                sequence[chunk_start:chunk_end]
            )
            corrected_chunks.append(corrected_chunk)
            attention_matrices.append(attention_matrix)
            confidence_chunks.append(confidences)
            if not eos_emitted:
                chunks_without_eos += 1

        if chunks_without_eos > 0:
            print(
                f"NOTE: {chunks_without_eos}/{len(offsets)} chunk(s) in this sequence did not emit "
                f"<EOS> within their decode budget; those chunks' output may be truncated/degraded. "
                f"Consider increasing decode_length_margin, or note this is expected on an "
                f"early-training checkpoint."
            )

        stitched_result = self._stitch_chunks(
            offsets, corrected_chunks, confidence_chunks
        )
        assert len(stitched_result) == 3, "Expected 3 return values since confidence_chunks was provided."
        final_sequence, corrected_offsets, per_base_confidence = stitched_result  # type: ignore

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        metrics = self._compute_metrics(sequence, final_sequence, len(offsets), latency_ms)
        metrics["chunks_without_eos"] = chunks_without_eos
        
        attention_chunks = [
            {
                "chunk_index": i,
                "corrected_start": corr_start,
                "corrected_end": corr_end,
                "source_start": src_start,
                "source_end": src_end,
                "attention_matrix": attn,
            }
            for i, ((src_start, src_end), (corr_start, corr_end), attn) in enumerate(
                zip(offsets, corrected_offsets, attention_matrices)
            )
        ]

        return {
            "corrected_sequence": final_sequence,
            "metrics": metrics,
            "attention_chunks": attention_chunks,
            "per_base_confidence": per_base_confidence,
        }


def parse_fasta_upload(raw_bytes: bytes) -> str:
    """
    Parses raw bytes from an uploaded .fasta/.fa file into a single raw
    nucleotide sequence string. Rejects multi-record files with a clear
    error (same single-record policy as data/simulator.py.load_reference())
    rather than silently using only the first record or concatenating
    unrelated contigs together. backend/main.py's file-upload endpoint is
    expected to translate a ValueError here into an HTTP 400.
    """
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Uploaded file is not valid UTF-8 text: {e}") from e

    header_count = 0
    seq_parts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header_count += 1
            if header_count > 1:
                raise ValueError(
                    "Uploaded FASTA file contains more than one record. This endpoint "
                    "corrects a single sequence at a time -- split multi-record files "
                    "and upload one record per request."
                )
            continue
        seq_parts.append(line)

    if header_count == 0:
        raise ValueError("Uploaded file has no FASTA header ('>...') line.")

    sequence = "".join(seq_parts).upper()
    if len(sequence) == 0:
        raise ValueError("Uploaded FASTA file has a header but no sequence data.")

    return sequence


def save_checkpoint(model: SequenceTranslationModel, checkpoint_path: str) -> None:
    """
    Companion to InferenceEngine.load_from_checkpoint -- writes the exact
    format that loader expects. training/train.py should call this rather
    than rolling its own torch.save(), so writer and reader can never drift.

    Writes to a temp file in the same directory, then atomically renames it
    into place (os.replace/Path.replace are atomic on POSIX, which is what
    Kaggle/Colab run on). This matters specifically because Kaggle's 12-hour
    session limit is a hard kill, not a graceful shutdown -- without this, a
    kill mid-write could leave a truncated, unloadable  checkpoint file behind
    with no warning until the next session tries to load it.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state_dict": model.state_dict(), "model_config": model.cfg}, tmp_path)
    tmp_path.replace(path)


def save_training_state(
    model: SequenceTranslationModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    path: str,
    scaler: Optional[GradScaler] = None,
) -> None:
    """
    Full training-resume checkpoint: model weights + optimizer state (Adam's
    momentum/variance estimates) + epoch + global_step (drives the
    teacher-forcing decay schedule) + best_val_loss (drives the "only
    checkpoint if it's the best" decision in save_checkpoint above).

    Deliberately a SEPARATE file/format from save_checkpoint's -- that one is
    the lightweight inference-only contract relied on by evaluate.py, the
    API, and benchmarking/correct_fastq.py, and must not drift. This one
    exists purely so training/train.py can resume a session that was cut off
    mid-run (e.g. Kaggle's 12-hour hard kill) without losing optimizer
    momentum or restarting the teacher-forcing schedule from the beginning --
    both of which would otherwise cause a real, if temporary, quality dip
    right after every resume. Also atomic-written, for the same reason as
    save_checkpoint.

    scaler: if training uses mixed precision (torch.cuda.amp.GradScaler),
    pass it here so its internal state (current loss-scale factor, growth
    tracker) is preserved across a resume too. Without this, a resume would
    still work correctly -- GradScaler re-adapts its scale within a handful
    of steps regardless -- but would silently discard a few steps' worth of
    scale-tuning progress. If scaler is None, this key is simply omitted
    (not written as None), so old training-state files written before AMP
    support existed and files written without a scaler stay unambiguous.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, tmp_path)
    tmp_path.replace(out_path)

def load_training_state(
    model: SequenceTranslationModel,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> Tuple[int, int, float]:
    """
    Loads a training-resume checkpoint written by save_training_state(),
    restoring model AND optimizer state in place. Returns
    (epoch, global_step, best_val_loss) so the caller's training loop can
    pick up exactly where it left off -- same epoch, same point in the
    teacher-forcing schedule, same "best so far" comparison baseline.

    scaler: if given AND the checkpoint contains a saved scaler state (i.e.
    it was written by a run that used AMP), restores it in place too.
    Checkpoints written before AMP support existed, or by a run that had
    AMP disabled, simply won't have a "scaler_state_dict" key -- handled
    gracefully (skipped, not an error), since a resumed run's GradScaler
    will just start from its default scale and re-adapt within a few steps,
    which is a minor efficiency point, not a correctness one.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["epoch"], checkpoint["global_step"], checkpoint["best_val_loss"]


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    import tempfile

    torch.manual_seed(0)
    random.seed(0)

    model = SequenceTranslationModel(SequenceTranslationConfig())
    engine = InferenceEngine(model, chunk_size=512, chunk_overlap=128, device=torch.device("cpu"))

    # -- 1. Checkpoint round-trip: save, load, verify identical predictions ---
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "model.pt")
        save_checkpoint(model, ckpt_path)
        loaded_engine = InferenceEngine.load_from_checkpoint(
            ckpt_path, device=torch.device("cpu"), chunk_size=512, chunk_overlap=128
        )

        test_seq = "".join(random.choice("ACGT") for _ in range(100))
        torch.manual_seed(1)
        out_a = engine.correct_sequence(test_seq)
        torch.manual_seed(1)
        out_b = loaded_engine.correct_sequence(test_seq)
        assert out_a["corrected_sequence"] == out_b["corrected_sequence"], (
            "Loaded-checkpoint model produced different output than the original -- "
            "checkpoint round-trip is not faithful."
        )
    print("[1/5] Checkpoint save/load round-trip passed (identical predictions).")

    # -- 2. Single-chunk path (short sequence, no stitching needed) -----------
    short_seq = "".join(random.choice("ACGT") for _ in range(200))
    result = engine.correct_sequence(short_seq)
    assert len(result["attention_chunks"]) == 1
    assert result["attention_chunks"][0]["source_start"] == 0
    assert result["attention_chunks"][0]["source_end"] == 200
    assert result["metrics"]["input_length"] == 200
    assert result["metrics"]["num_chunks"] == 1
    print(
        f"[2/5] Single-chunk path passed. corrected_length={result['metrics']['corrected_length']}, "
        f"edit_distance={result['metrics']['edit_distance']} (untrained model -- large edit "
        f"distance is expected, this only proves the pipeline runs correctly end to end)."
    )

    # -- 3. Multi-chunk path on a REAL excerpt of the real E. coli genome, -----
    #       forcing chunking with a small chunk_size, verifying stitching ------
    #       produces one coherent, non-empty, boundary-consistent sequence -----
    with open("data/reference/ecoli_k12_mg1655.fasta") as f:
        lines = f.readlines()
    real_genome_excerpt = "".join(l.strip() for l in lines[1:])[200_000:202_000]  # real 2000bp excerpt
    assert len(real_genome_excerpt) == 2000

    small_engine = InferenceEngine(model, chunk_size=512, chunk_overlap=128, device=torch.device("cpu"))
    offsets = small_engine._chunk_offsets(len(real_genome_excerpt))
    assert len(offsets) > 1, "Expected multi-chunk path to trigger for a 2000bp sequence at chunk_size=512"
    # verify contiguous coverage with exactly the expected overlap
    for (s0, e0), (s1, e1) in zip(offsets, offsets[1:]):
        assert s1 < e0, "Consecutive chunks must overlap"
    assert offsets[-1][1] == len(real_genome_excerpt)
    assert offsets[0][0] == 0

    result = small_engine.correct_sequence(real_genome_excerpt)
    assert len(result["attention_chunks"]) == len(offsets)
    # corrected_offsets must be contiguous and non-overlapping (each chunk's
    # contribution picks up exactly where the previous one left off)
    for chunk_info in result["attention_chunks"]:
        assert chunk_info["corrected_start"] <= chunk_info["corrected_end"]
    corr_starts = [c["corrected_start"] for c in result["attention_chunks"]]
    corr_ends = [c["corrected_end"] for c in result["attention_chunks"]]
    for i in range(len(corr_starts) - 1):
        assert corr_ends[i] == corr_starts[i + 1], (
            f"Stitched contributions must be contiguous with no gap/overlap: chunk {i} ends at "
            f"{corr_ends[i]}, chunk {i+1} starts at {corr_starts[i+1]}"
        )
    assert corr_ends[-1] == len(result["corrected_sequence"])
    print(
        f"[3/5] Multi-chunk path on a REAL 2000bp E. coli excerpt passed: {len(offsets)} chunks, "
        f"stitched contributions are contiguous with no gaps or overlaps. "
        f"corrected_length={result['metrics']['corrected_length']}, "
        f"latency_ms={result['metrics']['latency_ms']:.1f}"
    )

    # -- 4. Metrics sanity: identical sequences must have zero edit distance ---
    identical_metrics = InferenceEngine._compute_metrics("ACGTACGT", "ACGTACGT", num_chunks=1, latency_ms=1.0)
    assert identical_metrics["edit_distance"] == 0
    assert identical_metrics["num_matches"] == 8
    assert identical_metrics["num_substitutions"] == 0

    sub_metrics = InferenceEngine._compute_metrics("ACGTACGT", "ACGAACGT", num_chunks=1, latency_ms=1.0)
    assert sub_metrics["edit_distance"] == 1
    assert sub_metrics["num_substitutions"] == 1

    indel_metrics = InferenceEngine._compute_metrics("ACGTACGT", "ACGTAACGT", num_chunks=1, latency_ms=1.0)
    assert indel_metrics["edit_distance"] == 1
    assert indel_metrics["num_insertions"] == 1, indel_metrics  # corrected is LONGER than input -> insertion
    assert indel_metrics["num_deletions"] == 0, indel_metrics

    del_metrics = InferenceEngine._compute_metrics("ACGTAACGT", "ACGTACGT", num_chunks=1, latency_ms=1.0)
    assert del_metrics["edit_distance"] == 1
    assert del_metrics["num_deletions"] == 1, del_metrics  # corrected is SHORTER than input -> deletion
    assert del_metrics["num_insertions"] == 0, del_metrics
    print("[4/5] Metrics correctness (identical / substitution / insertion / deletion cases) passed.")

    # -- 5. FASTA-upload parsing ------------------------------------------------
    good_fasta = b">read1 some description\nACGT\nACGT\nAC\n"
    assert parse_fasta_upload(good_fasta) == "ACGTACGTAC"

    for bad_fasta, expected_substr in [
        (b"", "no FASTA header"),
        (b"ACGTACGT\n", "no FASTA header"),
        (b">r1\nACGT\n>r2\nACGT\n", "more than one record"),
        (b">r1\n", "no sequence data"),
        (b"\xff\xfe not valid utf-8", "not valid UTF-8"),
    ]:
        try:
            parse_fasta_upload(bad_fasta)
            raise AssertionError(f"Expected ValueError for input {bad_fasta!r}")
        except ValueError as e:
            assert expected_substr in str(e), f"Expected '{expected_substr}' in error, got: {e}"
    print("[5/5] parse_fasta_upload() validation passed.")

    print("\nAll inference_engine sanity checks passed.")