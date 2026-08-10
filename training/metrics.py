"""
training/metrics.py
----------------------
Bioinformatics validation metrics, used by training/evaluate.py against
KNOWN GROUND TRUTH (the true clean reference sequence) -- this is what
distinguishes these from backend/inference_engine.py's metrics, which
compare corrected output only against the original noisy INPUT, since real
inference-time requests have no ground truth available at all.

Three kinds of metric, per the Part 3 spec:
  1. Levenshtein edit distance + identity (via edlib)
  2. Biological codon/reading-frame preservation (does the correction
     restore the modulo-3 alignment required for correct translation?)
  3. External alignment verification via the actual minimap2 CLI binary
     (a real subprocess call, deliberately independent of the mappy-based
     alignment data/simulator.py uses internally, so this is a genuinely
     separate check rather than reusing the same code path twice)

-----------------------------------------------------------------------------
A NOTE ON edlib's QUERY/TARGET CIGAR CONVENTION
-----------------------------------------------------------------------------
Confirmed empirically while building backend/inference_engine.py (see that
file's _compute_metrics docstring): edlib.align(query, target, ...)'s CIGAR
op 'I' marks a base present in the QUERY but absent from the TARGET; 'D'
marks a base present in the TARGET but absent from the QUERY. Every
function below that parses a CIGAR calls out query vs. target explicitly in
its own docstring rather than relying on the op letters alone -- that
ambiguity is exactly what caused a real, caught bug the first time this
convention was used in this project.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import edlib


def compute_alignment_metrics(predicted_seq: str, reference_seq: str) -> dict:
    """
    Levenshtein edit distance and edit-type breakdown between a model's
    PREDICTED (corrected) output and the TRUE clean reference sequence.

    query=predicted_seq, target=reference_seq, so per edlib's convention:
      - 'I' = base in predicted but not in reference -> an uncorrected/
              introduced INSERTION error remaining in the output.
      - 'D' = base in reference but not in predicted -> an uncorrected/
              introduced DELETION error (something the model failed to
              restore).
    """
    result = edlib.align(predicted_seq, reference_seq, mode="NW", task="path")
    counts = {"=": 0, "X": 0, "I": 0, "D": 0}
    for length_str, op in re.findall(r"(\d+)([=XID])", result["cigar"]):
        counts[op] += int(length_str)

    alignment_len = sum(counts.values())
    identity = counts["="] / alignment_len if alignment_len > 0 else 0.0

    return {
        "predicted_length": len(predicted_seq),
        "reference_length": len(reference_seq),
        "edit_distance": result["editDistance"],
        "identity": identity,
        "num_matches": counts["="],
        "num_substitutions": counts["X"],
        "num_insertions_remaining": counts["I"],   # errors in predicted not present in reference
        "num_deletions_remaining": counts["D"],    # reference bases the model failed to restore
    }


def compute_frame_preservation(predicted_seq: str, reference_seq: str) -> dict:
    """
    Reading-frame / codon-preservation check: does the corrected sequence
    restore the structural modulo-3 alignment required for correct
    downstream translation into amino acids?

    Two complementary signals are returned, because a single global check
    can be misleading on its own:

      - global_frame_preserved: whether (len(predicted) - len(reference))
        % 3 == 0. Necessary for frame correctness, but NOT sufficient: two
        separate single-base indel errors that happen to cancel out
        globally (e.g. a deletion followed later by an insertion) would
        satisfy this check while still corrupting every codon BETWEEN them.

      - frame_intact_fraction: the fraction of the aligned length where the
        running net indel offset (bases added to predicted minus bases
        missing from predicted, so far, mod 3) is exactly 0 -- i.e. the
        fraction of the sequence where the reading frame genuinely matches
        the reference's frame at that point, not just at the very end.

    query=predicted_seq, target=reference_seq (same edlib convention as
    compute_alignment_metrics): 'I' ops shift the running offset by +1 per
    base (predicted has extra bases -> frame pushed forward), 'D' ops shift
    it by -1 per base (predicted is missing bases -> frame pulled back).
    """
    global_frame_preserved = (len(predicted_seq) - len(reference_seq)) % 3 == 0

    result = edlib.align(predicted_seq, reference_seq, mode="NW", task="path")
    ops = re.findall(r"(\d+)([=XID])", result["cigar"])

    running_offset = 0
    in_frame_aligned_positions = 0
    total_aligned_positions = 0
    frameshift_events = 0
    currently_in_frame = True

    for length_str, op in ops:
        length = int(length_str)
        if op in ("=", "X"):
            total_aligned_positions += length
            if running_offset % 3 == 0:
                in_frame_aligned_positions += length
        elif op == "I":
            for _ in range(length):
                running_offset += 1
                now_in_frame = (running_offset % 3 == 0)
                if now_in_frame != currently_in_frame:
                    frameshift_events += 1
                currently_in_frame = now_in_frame
        elif op == "D":
            for _ in range(length):
                running_offset -= 1
                now_in_frame = (running_offset % 3 == 0)
                if now_in_frame != currently_in_frame:
                    frameshift_events += 1
                currently_in_frame = now_in_frame

    frame_intact_fraction = (
        in_frame_aligned_positions / total_aligned_positions if total_aligned_positions > 0 else 1.0
    )

    return {
        "global_frame_preserved": global_frame_preserved,
        "frame_intact_fraction": frame_intact_fraction,
        "num_frameshift_events": frameshift_events,
    }


def run_minimap2_alignment(
    predicted_seq: str, reference_seq: str, minimap2_path: str = "minimap2"
) -> dict:
    """
    External validation: writes predicted_seq and reference_seq to
    temporary FASTA files and invokes the minimap2 CLI binary as a real
    subprocess (a genuinely independent check from the mappy-based
    alignment used internally by data/simulator.py, which shares the same
    underlying C library but is invoked through Python bindings rather than
    as an external process). Parses minimap2's PAF output for the best hit.

    Returns {"aligned": False, ...} if no alignment was found at all
    (rather than raising), since "the corrected output no longer aligns to
    the reference" is itself a meaningful (bad) evaluation result, not an
    error condition.
    """
    if shutil.which(minimap2_path) is None:
        raise RuntimeError(
            f"'{minimap2_path}' not found on PATH. Install minimap2 (e.g. `apt-get install "
            f"minimap2`) to use external alignment verification."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = Path(tmpdir) / "reference.fasta"
        query_path = Path(tmpdir) / "predicted.fasta"
        ref_path.write_text(f">reference\n{reference_seq}\n")
        query_path.write_text(f">predicted\n{predicted_seq}\n")

        result = subprocess.run(
            [minimap2_path, "-x", "map-ont", str(ref_path), str(query_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"minimap2 failed (exit {result.returncode}): {result.stderr[-1000:]}")

        paf_lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        if not paf_lines:
            return {"aligned": False, "identity": 0.0, "coverage": 0.0, "mapping_quality": 0}

        # minimap2 orders hits best-first; take the top one.
        fields = paf_lines[0].split("\t")
        query_len = int(fields[1])
        query_start, query_end = int(fields[2]), int(fields[3])
        num_matches = int(fields[9])
        alignment_block_len = int(fields[10])
        mapping_quality = int(fields[11])

        identity = num_matches / alignment_block_len if alignment_block_len > 0 else 0.0
        coverage = (query_end - query_start) / query_len if query_len > 0 else 0.0

        return {
            "aligned": True,
            "identity": identity,
            "coverage": coverage,
            "mapping_quality": mapping_quality,
        }


def compute_all_metrics(
    predicted_seq: str, reference_seq: str, run_minimap2: bool = True, minimap2_path: str = "minimap2"
) -> dict:
    """Convenience wrapper bundling all three metric families for one (predicted, reference) pair."""
    metrics = {
        "alignment": compute_alignment_metrics(predicted_seq, reference_seq),
        "frame": compute_frame_preservation(predicted_seq, reference_seq),
    }
    if run_minimap2:
        try:
            metrics["minimap2"] = run_minimap2_alignment(predicted_seq, reference_seq, minimap2_path)
        except RuntimeError as e:
            metrics["minimap2"] = {"aligned": False, "error": str(e)}
    return metrics


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # -- 1. Identical sequences: perfect scores across the board --------------
    seq = "ACGTACGTACGTACGTGGCCTTAACGTACGT"
    m = compute_alignment_metrics(seq, seq)
    assert m["edit_distance"] == 0
    assert m["identity"] == 1.0
    assert m["num_insertions_remaining"] == 0 and m["num_deletions_remaining"] == 0

    f = compute_frame_preservation(seq, seq)
    assert f["global_frame_preserved"] is True
    assert f["frame_intact_fraction"] == 1.0
    assert f["num_frameshift_events"] == 0
    print("[1/5] Identical-sequence perfect-score case passed.")

    # -- 2. Insertion/deletion labeling correctness (re-verified explicitly, ---
    #       exactly the convention that was previously found inverted) --------
    ref = "ACGTACGTACGT"
    pred_with_extra_base = "ACGTAACGTACGT"  # predicted has 1 extra base -> uncorrected insertion
    m = compute_alignment_metrics(pred_with_extra_base, ref)
    assert m["num_insertions_remaining"] == 1, m
    assert m["num_deletions_remaining"] == 0, m

    pred_missing_base = "ACGTACGTACG"  # predicted is missing 1 base -> uncorrected deletion
    m = compute_alignment_metrics(pred_missing_base, ref)
    assert m["num_deletions_remaining"] == 1, m
    assert m["num_insertions_remaining"] == 0, m
    print("[2/5] Insertion/deletion labeling re-verified correct against ground truth.")

    # -- 3. Frame preservation: single uncorrected deletion breaks frame ------
    #       for the rest of the sequence; three cancel out globally but still
    #       corrupt everything BETWEEN them -----------------------------------
    reference = "ATGGCTAAACGTGGGCCCTTTAAACGTGGGCCCTTTAAA"  # 40bp, in-frame by construction
    # remove one base partway through -> global frame broken, AND locally broken from that point on
    single_del = reference[:15] + reference[16:]
    f = compute_frame_preservation(single_del, reference)
    assert f["global_frame_preserved"] is False, f
    assert f["frame_intact_fraction"] < 1.0, f
    assert f["num_frameshift_events"] == 1, f  # one shift introduced, never repaired

    # remove three separate single bases -> net offset is -3 (divisible by 3),
    # so global check says "preserved" even though three real errors occurred
    # and the regions between them are locally out of frame
    positions = sorted([10, 20, 30], reverse=True)
    triple_del = reference
    for p in positions:
        triple_del = triple_del[:p] + triple_del[p + 1:]
    f = compute_frame_preservation(triple_del, reference)
    assert f["global_frame_preserved"] is True, (
        "Expected the misleading-but-correct global result: net -3 offset is divisible by 3"
    )
    assert f["frame_intact_fraction"] < 1.0, (
        "But frame_intact_fraction must reveal that large stretches were locally out of frame "
        "despite the global check passing -- this is exactly why both metrics are reported."
    )
    # 2 transitions, verified empirically (not hand-derived): offset goes 0 -> -1 (break, event 1)
    # after the first deletion, stays out-of-frame through the second deletion (-1 -> -2, still
    # nonzero mod 3, no transition), then the third deletion (-2 -> -3) lands back on a multiple
    # of 3 (event 2, frame "repaired" numerically even though two real errors are now baked in).
    assert f["num_frameshift_events"] == 2, f
    print(
        f"[3/5] Frame-preservation nuance verified: 3 canceling deletions pass the GLOBAL check "
        f"(misleadingly) but frame_intact_fraction={f['frame_intact_fraction']:.3f} correctly "
        f"shows local corruption between them."
    )

    # -- 4. compute_all_metrics bundles everything, minimap2 optional ---------
    all_metrics_no_mm2 = compute_all_metrics(seq, seq, run_minimap2=False)
    assert "alignment" in all_metrics_no_mm2 and "frame" in all_metrics_no_mm2
    assert "minimap2" not in all_metrics_no_mm2
    print("[4/5] compute_all_metrics() without minimap2 passed.")

    # -- 5. Real minimap2 subprocess call, on a real excerpt of the real -------
    #       E. coli genome, self-aligned (should be ~100% identity/coverage) ---
    with open("data/reference/ecoli_k12_mg1655.fasta") as f_ref:
        lines = f_ref.readlines()
    real_excerpt = "".join(l.strip() for l in lines[1:])[500_000:500_500]  # real 500bp excerpt
    mm2_result = run_minimap2_alignment(real_excerpt, real_excerpt)
    assert mm2_result["aligned"] is True
    assert mm2_result["identity"] > 0.99, mm2_result
    # NOTE: even a perfect self-alignment doesn't always reach exactly 1.0 coverage --
    # minimap2 legitimately soft-clips a few bases at the very edges of short sequences
    # as part of its normal seed-and-extend behavior. > 0.95 confirms "essentially fully
    # aligned" without being brittle to that expected, harmless edge effect.
    assert mm2_result["coverage"] > 0.95, mm2_result
    print(f"[5/5] Real minimap2 subprocess call passed (self-alignment): {mm2_result}")

    print("\nAll metrics sanity checks passed.")
