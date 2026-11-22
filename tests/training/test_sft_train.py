"""The SFT loop: step arithmetic, the schedule, accumulation and determinism."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from verifiable_reward_lab.training.sft_config import SFTConfig
from verifiable_reward_lab.training.sft_data import IGNORE_INDEX, SFTDataset, SFTExample
from verifiable_reward_lab.training.sft_train import (
    SFTResult,
    build_optimizer,
    evaluate_loss,
    fit,
    lr_at,
    memory_gb,
    total_steps,
    warmup_steps,
)

if TYPE_CHECKING:
    from torch import Tensor

VOCAB = 16
PAD_ID = 0
CPU = torch.device("cpu")


@dataclass
class TinyOutput:
    logits: Tensor


class TinyLM(nn.Module):
    """A four-tensor language model, enough to exercise every path in fit."""

    def __init__(self, vocab_size: int = VOCAB, d_model: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, *, input_ids: Tensor, attention_mask: Tensor) -> TinyOutput:
        del attention_mask  # a bag-of-tokens model has nothing to mask
        return TinyOutput(logits=self.head(self.embed(input_ids)))


def tiny_model(seed: int = 0) -> TinyLM:
    """A model whose initial weights are a function of `seed` alone."""
    torch.manual_seed(seed)
    return TinyLM()


def tiny_dataset(n: int = 4) -> SFTDataset:
    """n examples of different lengths, one prompt token each.

    Lengths differ on purpose. With every example the same length, each
    microbatch keeps the same number of labelled tokens, the accumulated
    gradient stops depending on how examples were grouped, and a test that
    two seeds train differently would pass on an implementation that never
    shuffled at all.
    """
    examples = tuple(
        SFTExample(
            input_ids=tuple(range(1, 4 + i)),
            labels=(IGNORE_INDEX, *range(2, 4 + i)),
            n_prompt_tokens=1,
        )
        for i in range(n)
    )
    return SFTDataset(examples=examples, n_dropped=0, max_seq_len=8)


def tiny_config(**kw: object) -> SFTConfig:
    """A config that trains in milliseconds and still steps and decays."""
    base: dict[str, object] = {
        "seed": 0,
        "epochs": 1,
        "batch_size": 2,
        "grad_accum": 2,
        "max_seq_len": 8,
        "lr": 1e-1,
        "warmup_ratio": 0.0,
        "device": "cpu",
    }
    return SFTConfig(**{**base, **kw})  # ty: ignore[invalid-argument-type]


def test_total_steps_matches_the_protocol_budget() -> None:
    """6973 examples at effective batch 16, two epochs. The number in section 7."""
    assert total_steps(6973, SFTConfig()) == 872


def test_a_trailing_partial_group_still_steps() -> None:
    """Five examples at batch 2 is three microbatches, not two steps' worth."""
    config = tiny_config()

    assert total_steps(5, config) == 2
    assert total_steps(4, config) == 1


def test_total_steps_rejects_an_empty_dataset() -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        total_steps(0, tiny_config())


def test_warmup_then_linear_decay_to_zero() -> None:
    config = tiny_config(lr=1e-3, warmup_ratio=0.1)
    total = 100
    warm = warmup_steps(total, config)

    assert warm == 10
    assert lr_at(0, total=total, config=config) == pytest.approx(config.lr / warm)
    assert lr_at(warm - 1, total=total, config=config) == pytest.approx(config.lr)

    schedule = [lr_at(s, total=total, config=config) for s in range(total)]

    assert max(schedule) == pytest.approx(config.lr)
    assert min(schedule) >= 0.0
    assert schedule[-1] < schedule[warm]
    assert schedule[-1] == pytest.approx(config.lr / (total - warm), rel=1e-6)


def test_no_warmup_starts_at_full_lr() -> None:
    config = tiny_config(lr=1e-3, warmup_ratio=0.0)

    assert lr_at(0, total=10, config=config) == pytest.approx(config.lr)


def test_optimizer_decays_matrices_only() -> None:
    config = tiny_config(weight_decay=0.1)
    optimizer = build_optimizer(tiny_model(), config)

    decayed, plain = optimizer.param_groups

    assert decayed["weight_decay"] == pytest.approx(0.1)
    assert plain["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decayed["params"])
    assert all(p.dim() < 2 for p in plain["params"])


def test_fit_takes_the_steps_it_promised() -> None:
    config = tiny_config(epochs=2)
    dataset = tiny_dataset(5)

    result = fit(
        tiny_model(),
        dataset,
        config=config,
        device=CPU,
        pad_id=PAD_ID,
        verbose=False,
    )

    assert isinstance(result, SFTResult)
    assert result.steps == total_steps(len(dataset), config)
    assert result.epochs == 2
    assert result.s_per_step > 0


def test_fit_lowers_the_loss() -> None:
    config = tiny_config(epochs=20)
    dataset = tiny_dataset()
    model = tiny_model()

    before = evaluate_loss(
        model, dataset, batch_size=config.batch_size, pad_id=PAD_ID, device=CPU
    )
    fit(model, dataset, config=config, device=CPU, pad_id=PAD_ID, verbose=False)
    after = evaluate_loss(
        model, dataset, batch_size=config.batch_size, pad_id=PAD_ID, device=CPU
    )

    assert after < before


def test_the_same_seed_gives_the_same_run() -> None:
    """Two runs of one config must be one number, or seed variability is
    measuring the implementation instead of the training."""
    config = tiny_config(epochs=3)

    losses = [
        fit(
            tiny_model(),
            tiny_dataset(),
            config=config,
            device=CPU,
            pad_id=PAD_ID,
            verbose=False,
        ).final_train_loss
        for _ in range(2)
    ]

    assert losses[0] == pytest.approx(losses[1])


def test_a_different_seed_gives_a_different_run() -> None:
    config = tiny_config(epochs=3)
    other = tiny_config(epochs=3, seed=1)

    first = fit(
        tiny_model(),
        tiny_dataset(),
        config=config,
        device=CPU,
        pad_id=PAD_ID,
        verbose=False,
    )
    second = fit(
        tiny_model(),
        tiny_dataset(),
        config=other,
        device=CPU,
        pad_id=PAD_ID,
        verbose=False,
    )

    assert first.final_train_loss != pytest.approx(second.final_train_loss)


def test_dev_loss_is_measured_and_the_model_goes_back_to_training() -> None:
    config = tiny_config(epochs=2)
    model = tiny_model()

    result = fit(
        model,
        tiny_dataset(),
        config=config,
        device=CPU,
        pad_id=PAD_ID,
        dev=tiny_dataset(2),
        eval_interval=1,
        verbose=False,
    )

    assert not math.isnan(result.final_dev_loss)
    assert result.best_step > 0
    assert model.training


def test_without_dev_the_loss_is_nan_not_zero() -> None:
    """A missing measurement must not read as a good one."""
    result = fit(
        tiny_model(),
        tiny_dataset(),
        config=tiny_config(),
        device=CPU,
        pad_id=PAD_ID,
        verbose=False,
    )

    assert math.isnan(result.final_dev_loss)
    assert math.isnan(result.best_dev_loss)
    assert result.best_step == -1


def test_metrics_are_written_one_record_per_evaluation(tmp_path: Path) -> None:
    config = tiny_config(epochs=2)
    path = tmp_path / "metrics.jsonl"

    result = fit(
        tiny_model(),
        tiny_dataset(),
        config=config,
        device=CPU,
        pad_id=PAD_ID,
        dev=tiny_dataset(2),
        eval_interval=1,
        metrics_path=path,
        verbose=False,
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(records) == result.steps
    assert [r["step"] for r in records] == list(range(1, result.steps + 1))
    assert records[-1]["epoch"] == 2
    assert all(r["lr"] > 0 for r in records)
    assert all(r["s_per_step"] > 0 for r in records)


def test_a_rerun_replaces_the_metrics_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("stale\n", encoding="utf-8")

    fit(
        tiny_model(),
        tiny_dataset(),
        config=tiny_config(),
        device=CPU,
        pad_id=PAD_ID,
        eval_interval=1,
        metrics_path=path,
        verbose=False,
    )

    assert "stale" not in path.read_text()


def test_dev_is_measured_less_often_than_the_log(tmp_path: Path) -> None:
    """The point of splitting the two cadences: a line every step, a dev
    evaluation only when one is affordable."""
    path = tmp_path / "metrics.jsonl"

    fit(
        tiny_model(),
        tiny_dataset(),
        config=tiny_config(epochs=4),
        device=CPU,
        pad_id=PAD_ID,
        dev=tiny_dataset(2),
        log_interval=1,
        eval_interval=2,
        metrics_path=path,
        verbose=False,
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(records) == 4
    # null rather than a repeat of the last measurement, so a reader can
    # tell the two apart
    assert [r["dev_loss"] is not None for r in records] == [False, True, False, True]
    assert all(r["train_loss"] > 0 for r in records)


def test_the_committed_config_keeps_the_protocol_budget() -> None:
    """The micro/accumulation split is an engineering choice; the effective
    batch and the step count are the protocol's. Changing the first must
    never move the second."""
    repo_root = Path(__file__).resolve().parents[2]
    config = SFTConfig.load(repo_root / "configs" / "sft_qwen05b.yaml")

    assert config.effective_batch == 16
    assert total_steps(6973, config) == 872


def test_memory_reporting_is_silent_off_mps() -> None:
    """The instrumentation must not decide whether the loop runs."""
    assert memory_gb(CPU) == (0.0, 0.0)


def test_the_padded_width_is_recorded(tmp_path: Path) -> None:
    """Which shape the accelerator was asked to compile for, per step.

    Bucketing is switched off so the recorded width is the raw batch
    maximum. `fit` buckets to 128 by default; bucketing itself is covered
    by `test_fit_compiles_for_a_handful_of_shapes`.
    """
    path = tmp_path / "metrics.jsonl"
    dataset = tiny_dataset()

    fit(
        tiny_model(),
        dataset,
        config=tiny_config(),
        device=CPU,
        pad_id=PAD_ID,
        pad_to_multiple_of=None,
        log_interval=1,
        metrics_path=path,
        verbose=False,
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert all(r["pad_width"] > 0 for r in records)
    assert max(r["pad_width"] for r in records) == max(
        len(e.input_ids) for e in dataset.examples
    )


def test_fit_compiles_for_a_handful_of_shapes(tmp_path: Path) -> None:
    """The fix for the MPS collapse: every step reports a bucketed width, so
    the accelerator sees a few shapes rather than one per batch."""
    path = tmp_path / "metrics.jsonl"

    fit(
        tiny_model(),
        tiny_dataset(8),
        config=tiny_config(epochs=3),
        device=CPU,
        pad_id=PAD_ID,
        pad_to_multiple_of=4,
        log_interval=1,
        metrics_path=path,
        verbose=False,
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    widths = {r["pad_width"] for r in records}

    assert all(w % 4 == 0 for w in widths)
    assert len(widths) <= 3
