"""
The SFT baseline run: the arm RQ1 compares GRPO against.

    uv run python -m verifiable_reward_lab.training.sft_train

Three choices here are not in `SFTConfig`, and are deliberate.

The schedule is linear warmup then linear decay to zero. The protocol froze
`warmup_ratio` and left the shape after it open; at 872 steps the gap between
linear and cosine is smaller than the spread across seeds, so the cheaper
description wins and is recorded rather than discovered later.

The weights saved are the LAST ones, not the best by dev loss. Two epochs is
what the protocol specifies; picking a checkpoint by dev loss would be a
selection rule, and an unfrozen selection rule is how a baseline quietly gets
tuned against the thing it is supposed to be a control for. The dev loss is
still measured and written down - if the second epoch hurts, that is a
finding to report, not a knob to turn.

The evaluation cadence lives in the function signature and in summary.json,
not in the config. It changes what is observed, never what is produced: the
same seed and config give the same weights whether dev is scored ten times
or never.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch

from verifiable_reward_lab.training.losses import causal_lm_loss
from verifiable_reward_lab.training.sft_data import Tokenizer, collate
from verifiable_reward_lab.training.trainer import resolve_device, set_seed

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor
    from torch.nn import Parameter

    from verifiable_reward_lab.training.sft_config import SFTConfig
    from verifiable_reward_lab.training.sft_data import SFTDataset


class Output(Protocol):
    """What this file needs from a forward pass."""

    logits: Tensor


class CausalLM(Protocol):
    """What this file needs from a model.

    A protocol rather than the transformers class so the loop can be tested
    in milliseconds against a model with four parameters, and so GRPO can
    hand its policy to the same code without inheriting anything.
    """

    training: bool

    def __call__(self, *, input_ids: Tensor, attention_mask: Tensor) -> Output: ...

    def parameters(self) -> Iterator[Parameter]: ...

    def train(self, mode: bool = ...) -> object: ...

    def eval(self) -> object: ...


@dataclass(frozen=True)
class SFTResult:
    """What one SFT run is, reduced to numbers. Written to summary.json."""

    steps: int
    epochs: int
    final_train_loss: float
    final_dev_loss: float
    best_dev_loss: float
    best_step: int
    """Where dev loss bottomed out. Reported, never used to pick weights."""

    wall_clock_s: float
    s_per_step: float
    """Protocol section 7. The figure the run budget is built from."""

    def to_dict(self) -> dict[str, float | int]:
        """Plain-data view, ready for json.dump."""
        return {
            "steps": self.steps,
            "epochs": self.epochs,
            "final_train_loss": self.final_train_loss,
            "final_dev_loss": self.final_dev_loss,
            "best_dev_loss": self.best_dev_loss,
            "best_step": self.best_step,
            "wall_clock_s": self.wall_clock_s,
            "s_per_step": self.s_per_step,
        }


def total_steps(n_examples: int, config: SFTConfig) -> int:
    """Optimiser steps the whole run will take.

    Args:
        n_examples: Examples that survived the length cap.
        config: Reads batch_size, grad_accum and epochs.

    Returns:
        Steps across every epoch. A trailing partial accumulation group
        still steps, so it counts.

    Raises:
        ValueError: There is nothing to train on.
    """
    if n_examples <= 0:
        msg = "cannot train on an empty dataset"
        raise ValueError(msg)

    micro = math.ceil(n_examples / config.batch_size)
    return math.ceil(micro / config.grad_accum) * config.epochs


def warmup_steps(total: int, config: SFTConfig) -> int:
    """How many steps the warmup covers, rounded to a whole step."""
    return round(config.warmup_ratio * total)


def lr_at(step: int, *, total: int, config: SFTConfig) -> float:
    """Learning rate for `step`: linear warmup, then linear decay to zero.

    Args:
        step: 0-indexed optimiser step.
        total: Steps in the whole run.
        config: Reads lr and warmup_ratio.

    Returns:
        The learning rate to set before this step.
    """
    warm = warmup_steps(total, config)

    # (step + 1) so the first step is not taken at lr 0. adam's second
    # moment is noise for the first few steps and a full-size update on a
    # noisy denominator is how a run diverges in the first minute.
    if warm > 0 and step < warm:
        return config.lr * (step + 1) / warm

    decay = max(1, total - warm)
    progress = min(1.0, max(0.0, (step - warm) / decay))
    return config.lr * (1.0 - progress)


def memory_gb(device: torch.device) -> tuple[float, float]:
    """(tensors, total held by the driver) in GiB, or (0, 0) off mps.

    The two differ by the compiled-graph cache and the allocator's free
    pools. A run whose tensor figure is flat while the driver figure climbs
    is not leaking tensors - it is compiling a new graph per shape, which
    is exactly what variable-length padding produces.
    """
    if device.type != "mps":
        return (0.0, 0.0)

    gib = 1024**3
    current = getattr(torch.mps, "current_allocated_memory", None)
    driver = getattr(torch.mps, "driver_allocated_memory", None)
    return (
        current() / gib if current else 0.0,
        driver() / gib if driver else 0.0,
    )


def build_optimizer(model: CausalLM, config: SFTConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only.

    Matmul weights and embeddings decay; biases and norm gains do not.
    Decaying a norm gain pulls the layer's ability to scale towards zero.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {
            "params": [p for p in params if p.dim() >= 2],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr)


def batch_loss(
    model: CausalLM,
    examples: list[Any],
    *,
    pad_id: int,
    device: torch.device,
    pad_to_multiple_of: int | None = None,
) -> Tensor:
    """Forward one microbatch and reduce it to a scalar loss."""
    batch = collate(examples, pad_id=pad_id, pad_to_multiple_of=pad_to_multiple_of)
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    return causal_lm_loss(out.logits, batch["labels"].to(device))


@torch.no_grad()
def evaluate_loss(
    model: CausalLM,
    dataset: SFTDataset,
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device,
    pad_to_multiple_of: int | None = None,
) -> float:
    """Mean answer-token loss over a dataset, in nats.

    Deterministic by construction: the whole set, in its frozen order, with
    no sampling. Cheap enough to run beside training, and unlike exact-match
    it says something about the second epoch while it is still running.
    """
    was_training = model.training
    model.eval()

    total = 0.0
    batches = 0
    for start in range(0, len(dataset), batch_size):
        chunk = list(dataset.examples[start : start + batch_size])
        total += batch_loss(
            model,
            chunk,
            pad_id=pad_id,
            device=device,
            pad_to_multiple_of=pad_to_multiple_of,
        ).item()
        batches += 1

    if was_training:
        model.train()
    return total / batches


def fit(
    model: CausalLM,
    dataset: SFTDataset,
    *,
    config: SFTConfig,
    device: torch.device,
    pad_id: int,
    dev: SFTDataset | None = None,
    pad_to_multiple_of: int | None = 128,
    log_interval: int = 1,
    eval_interval: int = 50,
    metrics_path: Path | None = None,
    verbose: bool = True,
) -> SFTResult:
    """Train `model` on `dataset` and return what the run measured.

    Does no I/O beyond `metrics_path`, loads nothing and saves no weights:
    `run` does that. Everything here is exercised by the tests against a
    four-parameter model.

    Args:
        model: Put into train mode and updated in place.
        dataset: The encoded training set.
        config: The frozen description of the run.
        device: Where to run.
        pad_id: Padding id for collate. Masked out either way.
        dev: Scored every eval_interval steps. None skips it.
        pad_to_multiple_of: Bucket the padded width, so the accelerator sees
            a handful of shapes instead of one per batch. See `collate`.
        log_interval: Steps between records. Cheap; defaults to every step.
        pad_to_multiple_of: Bucket the padded width. See `collate`.
        log_interval: Steps between metric records.
        eval_interval: Steps between dev evaluations. Expensive.
        metrics_path: One json record per log. None skips the file.
        verbose: Print one line per log.

    Returns:
        The run's SFTResult, s_per_step included.

    Raises:
        ValueError: The dataset is empty.
    """
    started = time.perf_counter()
    total = total_steps(len(dataset), config)

    set_seed(config.seed)
    optimizer = build_optimizer(model, config)
    model.train()

    if metrics_path is not None:
        metrics_path.write_text("", encoding="utf-8")  # a rerun starts a record

    step = 0
    window: list[float] = []
    widest = 0
    train_loss = math.nan
    dev_loss = math.nan
    best_dev = math.inf
    best_step = -1

    for epoch in range(config.epochs):
        # derived from the run seed, so a rerun sees the same order, and
        # epoch 2 is not epoch 1 again
        generator = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        micro = [
            order[i : i + config.batch_size]
            for i in range(0, len(order), config.batch_size)
        ]

        optimizer.zero_grad(set_to_none=True)

        for i, indices in enumerate(micro, start=1):
            chunk = [dataset.examples[j] for j in indices]
            # the width every example in this microbatch is padded to, and
            # therefore the shape the accelerator is asked to compile for
            raw = max(len(e.input_ids) for e in chunk)
            if pad_to_multiple_of:
                raw = math.ceil(raw / pad_to_multiple_of) * pad_to_multiple_of
            # the shape the accelerator was asked to compile for
            widest = max(widest, raw)
            loss = batch_loss(
                model,
                chunk,
                pad_id=pad_id,
                device=device,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            window.append(loss.item())

            # divided before backward: gradients accumulate as a sum, and
            # without this the effective learning rate is grad_accum times
            # the one the config names. note this makes the step a mean of
            # microbatch means - a microbatch of short answers weighs the
            # same as one of long answers. constant across every arm of the
            # study, so it cancels in the comparison.
            (loss / config.grad_accum).backward()

            is_last_micro = i == len(micro)
            if i % config.grad_accum and not is_last_micro:
                continue

            lr = lr_at(step, total=total, config=config)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # global norm, not per-parameter: one exploding layer should
            # scale the whole update down, not be flattened on its own
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            is_final = is_last_micro and epoch == config.epochs - 1
            if step % log_interval and not is_final:
                continue

            train_loss = sum(window) / len(window)
            window.clear()
            tensors_gb, driver_gb = memory_gb(device)

            # the two cadences are separate because their costs are. a log
            # line is a division; a dev evaluation is 200 examples, 25
            # forward passes, more per step than the training step itself.
            # logging every step is what makes a slow run distinguishable
            # from a hung one, and it has to be affordable to be default.
            measured = False
            if dev is not None and (step % eval_interval == 0 or is_final):
                dev_loss = evaluate_loss(
                    model,
                    dev,
                    batch_size=config.batch_size,
                    pad_id=pad_id,
                    device=device,
                    pad_to_multiple_of=pad_to_multiple_of,
                )
                measured = True

            elapsed = time.perf_counter() - started

            if metrics_path is not None:
                record = {
                    "step": step,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    # null, not the stale value: a reader must be able to
                    # tell a measurement from a repeat of the last one
                    "dev_loss": dev_loss if measured else None,
                    "lr": lr,
                    "elapsed_s": round(elapsed, 3),
                    "s_per_step": round(elapsed / step, 4),
                    "pad_width": widest,
                    "tensors_gb": round(tensors_gb, 2),
                    "driver_gb": round(driver_gb, 2),
                }
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

            if verbose:
                shown = f"{dev_loss:.4f}" if measured else "     -"
                print(
                    f"step {step:>5}/{total} | epoch {epoch + 1} | "
                    f"train {train_loss:.4f} | dev {shown} | "
                    f"lr {lr:.2e} | pad {widest:>3} | "
                    f"mem {tensors_gb:4.1f}/{driver_gb:4.1f} GB | {elapsed:6.1f}s"
                )

            widest = 0

            if measured and dev_loss < best_dev:
                best_dev, best_step = dev_loss, step

    wall = time.perf_counter() - started
    return SFTResult(
        steps=step,
        epochs=config.epochs,
        final_train_loss=train_loss,
        final_dev_loss=dev_loss,
        best_dev_loss=best_dev if best_step > 0 else math.nan,
        best_step=best_step,
        wall_clock_s=wall,
        s_per_step=wall / step,
    )


class _HFModel(CausalLM, Protocol):
    """The slice of a transformers model `run` needs beyond the loop's."""

    config: Any

    def to(self, device: torch.device) -> object: ...

    def gradient_checkpointing_enable(self) -> object: ...

    def save_pretrained(self, path: str) -> object: ...


class _HFTokenizer(Tokenizer, Protocol):
    """The slice of a transformers tokenizer `run` needs beyond encoding."""

    def save_pretrained(self, path: str) -> object: ...


def run(
    config: SFTConfig,
    *,
    out_dir: Path,
    checkpoint_dir: Path,
    data_dir: Path | None = None,
    gradient_checkpointing: bool = True,
    pad_to_multiple_of: int | None = 128,
    log_interval: int = 1,
    eval_interval: int = 50,
    n_dev: int = 200,
    verbose: bool = True,
) -> SFTResult:
    """Train the SFT baseline and write the run to disk.

    `out_dir` gets config.yaml, metrics.jsonl and summary.json - the record.
    `checkpoint_dir` gets the weights, and is not under runs/ on purpose:
    runs/ is evidence a reader can inspect, and 0.5B of weights is not.

    Args:
        config: The frozen SFT config.
        out_dir: Where the record goes.
        checkpoint_dir: Where the weights go.
        data_dir: Override the corpus directory.
        gradient_checkpointing: Recompute activations in the backward pass
            instead of storing them. Costs roughly a third more compute and
            is what makes the run fit at all; the weights are unchanged.
        pad_to_multiple_of: Bucket the padded width. See `collate`.
        log_interval: Steps between metric records.
        eval_interval: Steps between dev evaluations.
        n_dev: Dev examples scored. 200 is the prefix the metric uses.
        verbose: Print one line per evaluation.

    Returns:
        The run's SFTResult.

    Raises:
        ValueError: The tokenizer decorates prompts, or the data is missing.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from verifiable_reward_lab.data.gsm8k import load
    from verifiable_reward_lab.training.sft_data import (
        build_dataset,
        check_no_special_tokens,
    )

    device = resolve_device(config.device)

    tokenizer = cast(
        "_HFTokenizer",
        AutoTokenizer.from_pretrained(
            config.model_name, revision=config.model_revision
        ),
    )
    check_no_special_tokens(tokenizer, "What is two plus two?")

    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        msg = f"{config.model_name} has no eos token to pad with"
        raise ValueError(msg)

    split = (
        load(data_dir, seed=config.split_seed)
        if data_dir
        else load(seed=config.split_seed)
    )
    train_set = build_dataset(split.train, tokenizer, max_seq_len=config.max_seq_len)
    dev_set = build_dataset(
        split.dev[:n_dev], tokenizer, max_seq_len=config.max_seq_len
    )

    model = cast(
        "_HFModel",
        AutoModelForCausalLM.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            dtype=getattr(torch, config.dtype),
        ),
    )
    model.to(device)

    if gradient_checkpointing:
        # use_cache and checkpointing are mutually exclusive: the cache
        # keeps the activations checkpointing exists to throw away, and
        # transformers silently disables one of them if both are asked for
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / "config.yaml")

    if verbose:
        print(
            f"{config.model_name} on {device} | "
            f"train {len(train_set)} (+{train_set.n_dropped} dropped) | "
            f"dev {len(dev_set)} | {total_steps(len(train_set), config)} steps"
        )

    result = fit(
        model,
        train_set,
        config=config,
        device=device,
        pad_id=pad_id,
        dev=dev_set,
        pad_to_multiple_of=pad_to_multiple_of,
        log_interval=log_interval,
        eval_interval=eval_interval,
        metrics_path=out_dir / "metrics.jsonl",
        verbose=verbose,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # use_cache was turned off for checkpointing, and save_pretrained writes
    # the live config. left off, it is baked into the checkpoint, and every
    # later generation - the eval, every GRPO rollout - runs without a KV
    # cache and quadratically slower, with no error to say why.
    model.config.use_cache = True
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    summary: dict[str, Any] = {
        **result.to_dict(),
        "config": config.to_dict(),
        "checkpoint_dir": str(checkpoint_dir),
        "dev_fingerprint": split.fingerprint,
        "n_train": len(train_set),
        "n_train_dropped": train_set.n_dropped,
        "n_dev": len(dev_set),
        "pad_to_multiple_of": pad_to_multiple_of,
        "log_interval": log_interval,
        "eval_interval": eval_interval,
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Train one SFT baseline from a committed config."""
    from verifiable_reward_lab.training.sft_config import SFTConfig

    parser = argparse.ArgumentParser(description="train the SFT baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/sft_qwen05b.yaml"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--no-grad-checkpointing", action="store_true", help="store activations"
    )
    parser.add_argument("--pad-multiple", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--n-dev", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        config = SFTConfig.load(args.config)
        out_dir = args.out_dir or Path(f"runs/sft/seed-{config.seed}")
        checkpoint_dir = args.checkpoint_dir or Path(
            f"checkpoints/sft-seed-{config.seed}"
        )

        result = run(
            config,
            out_dir=out_dir,
            checkpoint_dir=checkpoint_dir,
            data_dir=args.data_dir,
            gradient_checkpointing=not args.no_grad_checkpointing,
            pad_to_multiple_of=args.pad_multiple,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            n_dev=args.n_dev,
        )
    except (ValueError, OSError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print(f"steps                  {result.steps}")
    print(f"s/step                 {result.s_per_step:.3f}")
    print(f"wall clock             {result.wall_clock_s / 60:.1f} min")
    print(f"final train loss       {result.final_train_loss:.4f}")
    print(f"final dev loss         {result.final_dev_loss:.4f}")
    print(
        f"best dev loss          {result.best_dev_loss:.4f}  (step {result.best_step})"
    )
    print(f"weights                {checkpoint_dir}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
