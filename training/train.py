"""
training/train.py
--------------------
The training loop: loads simulated (noisy, clean) pairs produced by
data/simulator.py, trains SequenceTranslationModel with a decaying
teacher-forcing schedule, and checkpoints the best model via
backend.inference_engine.save_checkpoint (never a raw torch.save -- see
that function's docstring for why).

-----------------------------------------------------------------------------
WHY THE TRAIN/VAL SPLIT HAPPENS BEFORE CHUNKING, NOT AFTER
-----------------------------------------------------------------------------
data/dataset.py.GenomeCorrectionDataset flattens whole reads into many
overlapping-free chunks. If the train/val split were done on those chunks
directly, two chunks from the SAME original read could end up on opposite
sides of the split -- and since Badread reads are simulated from a real
reference genome, a "held-out" validation chunk could come from a genomic
region the model saw (in a different chunk, from the same read) during
training. That's leakage, and it would make validation metrics look better
than the model actually generalizes. The split here is performed on the
list of AlignedPair objects (whole reads) BEFORE any chunking happens, so
no read contributes to both sets.
"""

import argparse
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    PAD_IDX,
    DEFAULT_MODEL_CHECKPOINT_PATH,
    DEFAULT_TRAINING_DATA_JSONL_PATH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_VAL_FRACTION,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_TEACHER_FORCING_START,
    DEFAULT_TEACHER_FORCING_END,
    DEFAULT_TEACHER_FORCING_WARMUP_FRACTION,
    DEFAULT_LR_SCHEDULER_FACTOR,
    DEFAULT_LR_SCHEDULER_PATIENCE,
    DEFAULT_LR_SCHEDULER_MIN_LR,
)
from data.dataset import AlignedPair, create_dataloader, load_aligned_pairs_from_jsonl
from model.sequence_translation_model import SequenceTranslationConfig, SequenceTranslationModel
from backend.inference_engine import save_checkpoint, save_training_state, load_training_state


def split_aligned_pairs(
    pairs: List[AlignedPair], val_fraction: float, seed: int
) -> Tuple[List[AlignedPair], List[AlignedPair]]:
    """Splits AlignedPairs (whole reads) into train/val BEFORE chunking. See module docstring."""
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    num_val = int(len(shuffled) * val_fraction)
    val_pairs = shuffled[:num_val]
    train_pairs = shuffled[num_val:]
    return train_pairs, val_pairs


def compute_teacher_forcing_ratio(
    global_step: int, total_steps: int, start: float, end: float, warmup_fraction: float = 0.0
) -> float:
    """
    Linear decay from `start` to `end`, delayed by `warmup_fraction`: decay
    doesn't begin until that fraction of total_steps has elapsed, holding at
    `start` until then. Added after a real run showed decaying from step 0
    causes exposure-bias instability before the model has had real training.
    """
    if total_steps <= 0:
        return end
    raw_progress = min(max(global_step / total_steps, 0.0), 1.0)
    if raw_progress <= warmup_fraction:
        return start
    remaining_span = max(1.0 - warmup_fraction, 1e-8)
    progress = (raw_progress - warmup_fraction) / remaining_span
    return start + (end - start) * progress


def _run_validation(
    model: SequenceTranslationModel,
    val_loader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    """
    Validation loss under teacher forcing (ratio=1.0), matching the standard
    practice of comparing train/val loss on the same objective. This is a
    fast, PLUMBING-level check the training loop uses to pick a checkpoint;
    it is deliberately not the same thing as training/evaluate.py's full
    free-running biological validation (edit distance, frame preservation,
    minimap2), which is far more expensive and meant to be run separately/
    periodically, not every epoch.

    Runs under the same autocast setting as training (use_amp) purely for
    speed/memory consistency with the training forward pass -- there's no
    backward pass here, so no GradScaler is needed.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            with autocast(device_type=device.type, enabled=use_amp):
                output = model(
                    src_tokens=batch["src_tokens"].to(device),
                    src_lengths=batch["src_lengths"].to(device),
                    rle_base_ids=batch["rle_base_ids"].to(device),
                    rle_run_lengths=batch["rle_run_lengths"].to(device),
                    target_tokens=batch["target_tokens"].to(device),
                    teacher_forcing_ratio=1.0,
                )
                targets = batch["target_tokens"][:, 1:].to(device)
                loss = loss_fn(output.logits.reshape(-1, output.logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            total_batches += 1
    model.train()
    return total_loss / total_batches if total_batches > 0 else float("nan")


def train(
    train_jsonl_path: str = DEFAULT_TRAINING_DATA_JSONL_PATH,
    val_jsonl_path: Optional[str] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    chunk_size: int = 512,
    min_chunk_size: int = 16,
    max_chunk_size: Optional[int] = None,
    teacher_forcing_start: float = DEFAULT_TEACHER_FORCING_START,
    teacher_forcing_end: float = DEFAULT_TEACHER_FORCING_END,
    teacher_forcing_warmup_fraction: float = DEFAULT_TEACHER_FORCING_WARMUP_FRACTION,
    lr_scheduler_factor: float = DEFAULT_LR_SCHEDULER_FACTOR,
    lr_scheduler_patience: int = DEFAULT_LR_SCHEDULER_PATIENCE,
    lr_scheduler_min_lr: float = DEFAULT_LR_SCHEDULER_MIN_LR,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    checkpoint_path: str = DEFAULT_MODEL_CHECKPOINT_PATH,
    device: Optional[torch.device] = None,
    seed: int = 42,
    log_every: int = 10,
    max_steps: Optional[int] = None,
    resume_from: Optional[str] = None,
    training_state_path: Optional[str] = None,
    checkpoint_every_steps: int = 200,
    max_wall_time_seconds: Optional[float] = None,
    safety_margin_seconds: float = 300.0,
    use_amp: Optional[bool] = None,
) -> dict:
    """
    Full training run. Returns a history dict ({"train_losses": [...],
    "val_losses": [...]}) -- useful both for plotting in a report and for
    this file's own sanity checks (verifying loss actually decreases).

    Kaggle/Colab session-limit support:
    - resume_from: path to a training-state checkpoint (written by this
      function's own periodic saves below) to resume from -- restores model
      weights, optimizer state (Adam momentum/variance), epoch, and
      global_step (which drives the teacher-forcing decay schedule), so a
      resumed run continues smoothly rather than restarting the schedule and
      losing optimizer momentum.
    - training_state_path: where periodic training-state checkpoints are
      written. Defaults to f"{checkpoint_path}.training_state.pt" if not
      given. Separate from checkpoint_path (the lightweight, inference-only
      "best" checkpoint) -- see backend.inference_engine.save_training_state.
    - checkpoint_every_steps: how often (in optimizer steps, not epochs) to
      write an UNCONDITIONAL training-state checkpoint, regardless of
      whether loss improved. This matters specifically because a single
      epoch on a real dataset can easily run longer than a 12-hour Kaggle
      session; without step-level checkpointing, a hard kill mid-epoch loses
      that epoch's compute entirely, since the existing "save on best"
      checkpoint (below) only ever writes at epoch boundaries.
    - max_wall_time_seconds / safety_margin_seconds: if set, the loop
      proactively saves training state and returns cleanly once elapsed
      wall-clock time passes (max_wall_time_seconds - safety_margin_seconds)
      -- e.g. for a 12-hour Kaggle session, passing max_wall_time_seconds=
      12*3600 stops with 5 minutes (default margin) to spare, so the process
      exits on its own terms rather than being hard-killed mid-step or
      mid-write.

    NOTE on resume_from: this function itself NEVER auto-detects a prior
    training_state_path on disk -- resume_from must be explicitly passed.
    This is deliberate: an old training_state.pt sitting at that path from
    unrelated earlier experimentation should never silently hijack a fresh
    run. If you want "auto-resume if a local checkpoint already exists"
    behavior (e.g. to survive a notebook cell being re-executed), implement
    that check at the CALL SITE (see the Kaggle notebook's training cell),
    not here.

    Mixed precision (AMP):
    - use_amp: if None (default), auto-enabled when device is CUDA and
      disabled on CPU (autocast/GradScaler are CUDA-oriented; forcing them
      on CPU would just add overhead for no benefit, and this codebase's own
      CPU smoke test below relies on plain fp32 numerics). Pass True/False
      to override the auto-detection explicitly -- e.g. --no-amp on the CLI
      forces it off even on a CUDA device, useful if you ever need to rule
      out AMP while debugging a NaN/instability issue.
    - Implemented as: forward pass + loss under autocast(), backward/step
      via GradScaler (which scales the loss up before backward to avoid
      fp16 gradient underflow, then unscales before the optimizer step).
      Gradient clipping runs on UNSCALED gradients (scaler.unscale_() is
      called before clip_grad_norm_) -- clipping against scaled gradients
      would clip against the wrong magnitude entirely.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Explicit device visibility -- this used to be silent, which meant a
    # misconfigured CUDA install (e.g. a CPU-only torch wheel from a broken
    # pip environment) would silently fall back to CPU with zero indication
    # anything was wrong, other than "training is unexpectedly slow." Now
    # it's the first thing printed, not something you have to go debug
    # separately.
    if device.type == "cuda":
        print(f"Training on GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(
            "Training on CPU. If you have a GPU and expected this to use it, run "
            "`python -c \"import torch; print(torch.cuda.is_available())\"` -- if "
            "that prints False, your PyTorch install likely doesn't have CUDA "
            "support (common after a broken/interrupted pip install); reinstall "
            "with the correct --index-url for your CUDA version."
        )

    if use_amp is None:
        use_amp = device.type == "cuda"
    elif use_amp and device.type != "cuda":
        print(
            "WARNING: use_amp=True was requested but device is not CUDA; autocast/GradScaler "
            "are CUDA-oriented and would add overhead with no benefit on CPU. Forcing AMP off."
        )
        use_amp = False
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    print(
        f"Config: batch_size={batch_size}, chunk_size={chunk_size}, epochs={num_epochs}. "
        f"NOTE: the decoder is autoregressive -- per-batch cost scales with chunk_size "
        f"(more sequential decode steps), largely independent of GPU vs CPU. If training "
        f"feels slow regardless of device, a smaller --chunk-size and/or larger "
        f"--batch-size (to amortize fixed per-step loop overhead) will help more than "
        f"switching hardware alone."
    )

    # -- data: split at the PAIR level, then build chunked datasets -----------
    if val_jsonl_path is not None:
        train_pairs = load_aligned_pairs_from_jsonl(train_jsonl_path)
        val_pairs = load_aligned_pairs_from_jsonl(val_jsonl_path)
    else:
        all_pairs = load_aligned_pairs_from_jsonl(train_jsonl_path)
        train_pairs, val_pairs = split_aligned_pairs(all_pairs, val_fraction, seed)

    print(f"Training pairs (reads): {len(train_pairs)} | Validation pairs (reads): {len(val_pairs)}")

    train_loader = create_dataloader(
        train_pairs, chunk_size=chunk_size, min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size, batch_size=batch_size, shuffle=True,
    )
    val_loader = (
        create_dataloader(
            val_pairs, chunk_size=chunk_size, min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size, batch_size=batch_size, shuffle=False,
        )
        if val_pairs
        else None
    )
    if val_loader is None:
        print("WARNING: no validation pairs available; checkpointing will use train loss instead of val loss.")

    # -- model / optimizer -----------------------------------------------------
    model = SequenceTranslationModel(SequenceTranslationConfig()).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_scheduler_factor,
        patience=lr_scheduler_patience, min_lr=lr_scheduler_min_lr,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = GradScaler(device=device.type, enabled=use_amp)

    training_state_path = training_state_path or f"{checkpoint_path}.training_state.pt"

    total_steps = num_epochs * max(len(train_loader), 1)
    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 0
    history = {"train_losses": [], "val_losses": []}

    if resume_from is not None:
        if not Path(resume_from).exists():
            raise FileNotFoundError(
                f"--resume-from was given but no file exists at {resume_from}. "
                f"If this is the first session, omit --resume-from; it's only for "
                f"continuing a previously-interrupted run."
            )
        resumed_epoch, global_step, best_val_loss = load_training_state(
            model, optimizer, resume_from, device, scaler=scaler
        )
        start_epoch = resumed_epoch  # that epoch was in progress or just completed; redo/continue it
        print(
            f"Resumed from {resume_from}: epoch={resumed_epoch}, global_step={global_step}, "
            f"best_val_loss={best_val_loss:.4f}. Continuing from epoch {start_epoch + 1}/{num_epochs}."
        )

    run_start_time = time.perf_counter()

    if max_steps is not None:
        print(f"DIAGNOSTIC MODE: stopping after {max_steps} steps (timing only, not a real training run).")
    if max_wall_time_seconds is not None:
        print(
            f"Wall-time budget: will save and stop cleanly after "
            f"{max_wall_time_seconds - safety_margin_seconds:.0f}s elapsed "
            f"({max_wall_time_seconds:.0f}s budget minus {safety_margin_seconds:.0f}s safety margin)."
        )

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.perf_counter()
        running_loss = 0.0
        running_batches = 0

        for batch in train_loader:
            if max_steps is not None and global_step >= max_steps:
                break

            step_start = time.perf_counter()
            teacher_forcing_ratio = compute_teacher_forcing_ratio(
                global_step, total_steps, teacher_forcing_start, teacher_forcing_end,
                warmup_fraction=teacher_forcing_warmup_fraction,
            )

            with autocast(device_type=device.type, enabled=use_amp):
                output = model(
                    src_tokens=batch["src_tokens"].to(device),
                    src_lengths=batch["src_lengths"].to(device),
                    rle_base_ids=batch["rle_base_ids"].to(device),
                    rle_run_lengths=batch["rle_run_lengths"].to(device),
                    target_tokens=batch["target_tokens"].to(device),
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )
                targets = batch["target_tokens"][:, 1:].to(device)
                loss = loss_fn(output.logits.reshape(-1, output.logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # Gradients must be unscaled BEFORE clipping -- clipping against
            # still-scaled gradients would clip against the wrong magnitude
            # entirely (the scale factor can be in the thousands).
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_batches += 1
            global_step += 1

            if checkpoint_every_steps and global_step % checkpoint_every_steps == 0:
                save_training_state(
                    model, optimizer, epoch, global_step, best_val_loss, training_state_path,
                    scaler=scaler, scheduler=scheduler,
                )

            elapsed = time.perf_counter() - run_start_time
            if max_wall_time_seconds is not None and elapsed >= (max_wall_time_seconds - safety_margin_seconds):
                save_training_state(
                    model, optimizer, epoch, global_step, best_val_loss, training_state_path,
                    scaler=scaler, scheduler=scheduler,
                )
                avg_loss = running_loss / running_batches if running_batches > 0 else float("nan")
                print(
                    f"\nWall-time budget reached at {elapsed:.0f}s (epoch {epoch+1}/{num_epochs}, "
                    f"step {global_step}/{total_steps}, avg loss this epoch so far={avg_loss:.4f}). "
                    f"Training state saved to {training_state_path} -- resume with --resume-from "
                    f"{training_state_path} in the next session."
                )
                history["train_losses"].append(avg_loss)
                return history

            if max_steps is not None:
                print(f"step {global_step}/{max_steps} took {time.perf_counter() - step_start:.3f}s "
                      f"loss={loss.item():.4f} "
                      f"src_len={batch['src_tokens'].size(1)} tgt_len={batch['target_tokens'].size(1)}")
            elif log_every and global_step % log_every == 0:
                print(
                    f"epoch {epoch+1}/{num_epochs} step {global_step}/{total_steps} "
                    f"loss={loss.item():.4f} teacher_forcing_ratio={teacher_forcing_ratio:.2f}"
                )

        if max_steps is not None and global_step >= max_steps:
            print(f"DIAGNOSTIC MODE: reached {max_steps} steps, stopping (no checkpoint saved).")
            history = {"train_losses": [running_loss / running_batches], "val_losses": []}
            return history

        train_loss = running_loss / running_batches if running_batches > 0 else float("nan")
        history["train_losses"].append(train_loss)

        if val_loader is not None:
            val_loss = _run_validation(model, val_loader, loss_fn, device, use_amp=use_amp)
            history["val_losses"].append(val_loss)
            checkpoint_metric = val_loss
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(checkpoint_metric)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                print(f"LR reduced: {prev_lr:.2e} -> {new_lr:.2e} (no improvement)")
            
        else:
            val_loss = None
            checkpoint_metric = train_loss
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(checkpoint_metric)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                print(f"LR reduced: {prev_lr:.2e} -> {new_lr:.2e} (no improvement)")

        epoch_time = time.perf_counter() - epoch_start
        val_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        print(f"== epoch {epoch+1}/{num_epochs} done in {epoch_time:.1f}s: train_loss={train_loss:.4f}{val_str} ==")

        is_new_best = checkpoint_metric < best_val_loss
        if is_new_best:
            best_val_loss = checkpoint_metric

        # Unconditional -- every epoch, regardless of whether it was the best
        # -- so a resume always has a state at most one epoch stale, on top
        # of the more frequent step-count-based saves above. best_val_loss is
        # already updated above at this point, so a resume's "best so far"
        # comparison baseline is always correct, never one epoch behind.
        save_training_state(
            model, optimizer, epoch + 1, global_step, best_val_loss, training_state_path, scaler=scaler, scheduler=scheduler,
        )

        if is_new_best:
            save_checkpoint(model, checkpoint_path)
            print(f"New best ({checkpoint_metric:.4f}) -- checkpoint saved to {checkpoint_path}")

    return history


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SequenceTranslationModel on simulated ONT pairs.")
    parser.add_argument("--train-data", default=DEFAULT_TRAINING_DATA_JSONL_PATH)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--min-chunk-size", type=int, default=16)
    parser.add_argument("--max-chunk-size", type=int, default=None)
    parser.add_argument("--checkpoint-path", default=DEFAULT_MODEL_CHECKPOINT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10,
                         help="Print a progress line every N optimizer steps. Raise this on "
                              "platforms with limited log buffers (e.g. Kaggle) to cut log volume.")
    parser.add_argument("--max-steps", type=int, default=None,
                         help="Stop after N steps and print per-step timing. Diagnostic only, no checkpoint saved.")
    parser.add_argument("--resume-from", default=None,
                         help="Path to a training-state checkpoint to resume from (see save_training_state).")
    parser.add_argument("--training-state-path", default=None,
                         help="Where periodic training-state checkpoints are written. "
                              "Defaults to '<checkpoint-path>.training_state.pt'.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=200,
                         help="Write an unconditional training-state checkpoint every N optimizer steps.")
    parser.add_argument("--max-wall-time-seconds", type=float, default=None,
                         help="Proactively save and stop cleanly after this many seconds elapsed "
                              "(minus --safety-margin-seconds). E.g. 43200 for a 12-hour Kaggle session.")
    parser.add_argument("--safety-margin-seconds", type=float, default=300.0,
                         help="Safety margin subtracted from --max-wall-time-seconds before stopping.")
    parser.add_argument("--no-amp", action="store_true",
                         help="Disable mixed-precision (AMP) training even on a CUDA device. "
                              "AMP is auto-enabled on CUDA and auto-disabled on CPU by default.")
    parser.add_argument("--teacher-forcing-start", type=float, default=DEFAULT_TEACHER_FORCING_START)
    parser.add_argument("--teacher-forcing-end", type=float, default=DEFAULT_TEACHER_FORCING_END)
    parser.add_argument("--teacher-forcing-warmup-fraction", type=float, default=DEFAULT_TEACHER_FORCING_WARMUP_FRACTION)
    parser.add_argument("--lr-scheduler-factor", type=float, default=DEFAULT_LR_SCHEDULER_FACTOR)
    parser.add_argument("--lr-scheduler-patience", type=int, default=DEFAULT_LR_SCHEDULER_PATIENCE)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=DEFAULT_LR_SCHEDULER_MIN_LR)
    return parser


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        args = _build_arg_parser().parse_args()
        train(
            train_jsonl_path=args.train_data,
            val_jsonl_path=args.val_data,
            val_fraction=args.val_fraction,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            chunk_size=args.chunk_size,
            min_chunk_size=args.min_chunk_size,
            max_chunk_size=args.max_chunk_size,
            checkpoint_path=args.checkpoint_path,
            seed=args.seed,
            log_every=args.log_every,
            max_steps=args.max_steps,
            resume_from=args.resume_from,
            training_state_path=args.training_state_path,
            checkpoint_every_steps=args.checkpoint_every_steps,
            max_wall_time_seconds=args.max_wall_time_seconds,
            safety_margin_seconds=args.safety_margin_seconds,
            use_amp=False if args.no_amp else None,
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # No CLI args: run a small, fast, REAL end-to-end training sanity
    # check -- real Badread reads from a real E. coli excerpt, real
    # minimap2-based breakpoints, a few real epochs, verifying loss
    # actually decreases (a genuine test of the training loop, not just
    # that it runs without crashing).
    #
    # SCALE NOTE: decode-loop cost is dominated by TARGET SEQUENCE LENGTH
    # (an inherently sequential, per-timestep Python loop -- see
    # model/rle_decoder.py), not batch size or chunk COUNT. A full-length
    # real Badread read (~3000bp on this reference) costs tens of seconds
    # per batch on CPU, which is fine for real training but far too slow
    # for a sanity check. So each real read used here is truncated down to
    # ~100bp via data.dataset._chunk_aligned_pair (reusing the exact same
    # alignment-safe cutting logic already tested in data/dataset.py,
    # applied here for speed rather than a full-genome max receptive
    # field) before being handed to train().
    # ------------------------------------------------------------------
    import json
    import tempfile

    from data.dataset import _chunk_aligned_pair
    from data.simulator import SimulatorConfig, simulate_training_data

    def _truncate_pair(pair: AlignedPair, max_len: int) -> AlignedPair:
        noisy_sub, clean_sub = _chunk_aligned_pair(pair, chunk_size=max_len, min_chunk_size=1)[0]
        return AlignedPair(noisy_sub, clean_sub, breakpoints=[(0, 0), (len(noisy_sub), len(clean_sub))])

    with tempfile.TemporaryDirectory() as tmpdir:
        with open("data/reference/ecoli_k12_mg1655.fasta") as f:
            lines = f.readlines()
        real_seq = "".join(l.strip() for l in lines[1:])
        excerpt_seq = real_seq[400_000:403_000]  # real 3kb excerpt

        excerpt_path = str(Path(tmpdir) / "excerpt.fasta")
        with open(excerpt_path, "w") as f:
            f.write(">excerpt\n")
            for i in range(0, len(excerpt_seq), 70):
                f.write(excerpt_seq[i:i + 70] + "\n")

        raw_jsonl_path = str(Path(tmpdir) / "raw_pairs.jsonl")
        stats = simulate_training_data(
            SimulatorConfig(reference_fasta=excerpt_path, output_jsonl=raw_jsonl_path, quantity="10x", seed=7)
        )
        assert stats["pairs_written"] >= 6, (
            f"Expected at least 6 real aligned pairs from the smoke-test excerpt, got "
            f"{stats['pairs_written']}."
        )
        print(f"[setup] Generated {stats['pairs_written']} real training pairs; using the first 6, truncated to ~100bp each.")

        raw_pairs = load_aligned_pairs_from_jsonl(raw_jsonl_path)
        small_pairs = [_truncate_pair(p, max_len=100) for p in raw_pairs[:6]]

        small_jsonl_path = str(Path(tmpdir) / "small_pairs.jsonl")
        with open(small_jsonl_path, "w") as f:
            for p in small_pairs:
                f.write(json.dumps({
                    "noisy_sequence": p.noisy_sequence,
                    "clean_sequence": p.clean_sequence,
                    "breakpoints": p.breakpoints,
                }) + "\n")

        checkpoint_path = str(Path(tmpdir) / "smoke_test_checkpoint.pt")
        history = train(
            train_jsonl_path=small_jsonl_path,
            val_fraction=0.3,          # ~4 train pairs, ~2 val pairs
            num_epochs=6,
            batch_size=4,
            learning_rate=2e-3,
            chunk_size=150,            # larger than any truncated pair -> exactly 1 chunk per pair
            min_chunk_size=1,
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu"),
            seed=7,
            log_every=1000,            # keep this smoke test's stdout short
        )

        assert len(history["train_losses"]) == 6
        assert all(torch.isfinite(torch.tensor(l)) for l in history["train_losses"]), history["train_losses"]
        print(f"\n[1/2] Training ran for 6 epochs with finite loss throughout: "
              f"{[round(l, 4) for l in history['train_losses']]}")

        first_loss = history["train_losses"][0]
        last_loss = history["train_losses"][-1]
        assert last_loss < first_loss, (
            f"Expected training loss to decrease over 6 epochs on this small real dataset "
            f"(first={first_loss:.4f}, last={last_loss:.4f}) -- if this fails, something in the "
            f"training loop (gradient flow, optimizer step, loss masking) is broken, not just slow."
        )
        print(f"[2/2] Loss decreased over training: {first_loss:.4f} -> {last_loss:.4f}")

        assert Path(checkpoint_path).exists(), "Expected a checkpoint to have been saved"
        print(f"\nCheckpoint saved and verified to exist at {checkpoint_path}")

    print("\nAll train.py sanity checks passed.")