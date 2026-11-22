"""
The SFT data path: one GSM8K example turned into input ids and loss labels.

The prompt is not rebuilt here. `evaluation.prompt.build_prompt` is the
frozen procedure, and training against a second copy of it is the quietest
way to invalidate RQ1 - two copies agree until someone edits one of them,
and nothing fails when they stop agreeing.

The tokenisation is split in two on purpose. `build_prompt` ends in
"Answer: ", with a trailing space, and BPE merges that space into the word
that follows it, so

    encode(prompt + answer) != encode(prompt) + encode(answer)

and they differ at exactly the position the loss mask has to land on.
Encoding the halves separately costs one extra call and buys two things: the
boundary is exact by construction rather than by arithmetic over a joint
encoding, and the prompt ids are the ids the evaluation harness feeds the
model, not ids that merely decode to the same string.

Answers lose GSM8K's `<<12*8=96>>` calculator annotations. They are an
artefact of how the corpus was written; kept, the model learns to emit them,
and every GRPO rollout then pays for tokens the verifier ignores.

Examples longer than `max_seq_len` are dropped, never truncated. Truncation
removes the trailing `#### <value>`, which is the only part of an answer the
verifier reads: a truncated example is a question with no answer, trained on
as though it had one. How many were dropped is a number the protocol wants
beside the other budget figures, so it travels with the dataset.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch

from verifiable_reward_lab.evaluation.prompt import build_prompt
from verifiable_reward_lab.training.losses import IGNORE_INDEX

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from verifiable_reward_lab.data.gsm8k import Example

N_SHOTS = 0
"""Frozen: the SFT model is trained, and evaluated, with no exemplars. It is
taught the format, so showing it four examples would measure the prompt."""

ANNOTATION = re.compile(r"<<[^>]*>>")


class Tokenizer(Protocol):
    """What this file needs from a tokenizer."""

    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...


@dataclass(frozen=True)
class SFTExample:
    """One encoded example: what the model reads, and what it is graded on.

    Plain ints rather than tensors so an example is a value - comparable,
    hashable, and cheap to keep around for the failure analysis. Batching
    turns them into tensors, once, in `collate`.
    """

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    """IGNORE_INDEX over the prompt, the answer's own ids after it."""

    n_prompt_tokens: int
    """Where the mask ends. Equal to len(encode(prompt)) by construction."""


@dataclass(frozen=True)
class SFTDataset:
    """The encoded training set, plus what building it cost."""

    examples: tuple[SFTExample, ...]
    n_dropped: int
    """Examples longer than max_seq_len. Reported, never silently zero."""

    max_seq_len: int

    def __len__(self) -> int:
        return len(self.examples)


def strip_annotations(answer: str) -> str:
    """Remove GSM8K's inline calculator annotations from an answer."""
    return ANNOTATION.sub("", answer)


def encode_example(example: Example, tokenizer: Tokenizer) -> SFTExample:
    """Encode one example into ids and labels.

    Args:
        example: A GSM8K item with its answer still carrying `#### <value>`.
        tokenizer: The model's tokenizer.

    Returns:
        The encoded example, masked over the prompt.

    Raises:
        ValueError: The tokenizer has no eos id to end the answer with.
    """
    eos = tokenizer.eos_token_id
    if eos is None:
        msg = "tokenizer has no eos_token_id; the model would never learn to stop"
        raise ValueError(msg)

    # add_special_tokens=False states what Qwen2's tokenizer does anyway
    # rather than relying on it: a tokenizer that prepends a bos here and
    # not in the eval harness would shift the mask by one and still train
    prompt_ids = tokenizer.encode(
        build_prompt(example.question, N_SHOTS), add_special_tokens=False
    )
    answer = strip_annotations(example.answer).rstrip()
    answer_ids = [
        *tokenizer.encode(answer, add_special_tokens=False),
        eos,  # graded, not masked: the stop is part of what is being learned
    ]

    return SFTExample(
        input_ids=tuple(prompt_ids + answer_ids),
        labels=tuple([IGNORE_INDEX] * len(prompt_ids) + answer_ids),
        n_prompt_tokens=len(prompt_ids),
    )


def build_dataset(
    examples: Sequence[Example], tokenizer: Tokenizer, *, max_seq_len: int
) -> SFTDataset:
    """Encode a split, dropping what does not fit.

    Args:
        examples: The training split, in the order it was frozen in.
        tokenizer: The model's tokenizer.
        max_seq_len: Prompt plus answer plus eos, in tokens.

    Returns:
        The encoded dataset, in input order, with the drop count.

    Raises:
        ValueError: max_seq_len is not positive.
    """
    if max_seq_len <= 0:
        msg = f"max_seq_len must be > 0, got {max_seq_len}"
        raise ValueError(msg)

    kept: list[SFTExample] = []
    dropped = 0
    for example in examples:
        encoded = encode_example(example, tokenizer)
        if len(encoded.input_ids) > max_seq_len:
            dropped += 1
            continue
        kept.append(encoded)

    return SFTDataset(examples=tuple(kept), n_dropped=dropped, max_seq_len=max_seq_len)


def collate(
    batch: Sequence[SFTExample], *, pad_id: int, pad_to_multiple_of: int | None = None
) -> dict[str, Tensor]:
    """Pad a batch to its longest example, or to the next bucket above it.

    Padding goes on the right, unlike generation, where `HFGenerator` puts it
    on the left. The two are not inconsistent: training reads every position
    in one pass and the attention mask says which are real, while generation
    continues from the right edge and would otherwise continue from padding.

    `pad_to_multiple_of` rounds the width up. Padding to the batch maximum
    gives a different (B, T) on almost every batch, and an accelerator that
    compiles per shape - Metal does - then compiles a graph per batch and
    caches every one of them. Measured on this model: tensor memory flat at
    5.6 GB while the driver climbed 14.8 -> 36.8 GB over five steps, and
    steps went from 9 s to 360 s on batches that were not even the widest.
    Rounding to 128 leaves five possible widths for a 640-token cap.

    It costs the compute of the padding it adds and changes nothing else:
    the added positions are already masked in the attention mask and carry
    IGNORE_INDEX in the labels, so the loss is identical.

    Args:
        batch: Encoded examples, at least one.
        pad_id: The id to pad input_ids with. Masked out either way.
        pad_to_multiple_of: Round the width up to a multiple of this. None
            pads to the longest example, which is what tests want and what
            a fixed-shape accelerator does not.

    Returns:
        input_ids, labels and attention_mask, each (B, T) and dtype long.

    Raises:
        ValueError: The batch is empty, or the bucket size is not positive.
    """
    if not batch:
        msg = "collate got an empty batch"
        raise ValueError(msg)

    if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
        msg = f"pad_to_multiple_of must be > 0, got {pad_to_multiple_of}"
        raise ValueError(msg)

    width = max(len(e.input_ids) for e in batch)
    if pad_to_multiple_of is not None:
        width = math.ceil(width / pad_to_multiple_of) * pad_to_multiple_of

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention: list[list[int]] = []
    for e in batch:
        pad = width - len(e.input_ids)
        input_ids.append([*e.input_ids, *([pad_id] * pad)])
        # padding is ignored twice over - by the mask and by the loss. the
        # second one is what matters: pad_id is eos on this tokenizer, and
        # a padded position left unmasked would teach the model to stop
        labels.append([*e.labels, *([IGNORE_INDEX] * pad)])
        attention.append([1] * len(e.input_ids) + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def check_no_special_tokens(tokenizer: Tokenizer, question: str) -> None:
    """Fail loudly if the tokenizer would add a token the harness does not.

    The masking here encodes with `add_special_tokens=False`, while the
    evaluation harness calls the tokenizer plainly and gets the default.
    Qwen2 adds nothing either way, which is why the two agree - but that is
    a property of one tokenizer, not a rule, and a bos appearing on one side
    only would shift every mask by one token and still train happily.

    Raises:
        ValueError: The tokenizer adds special tokens to a prompt.
    """
    prompt = build_prompt(question, N_SHOTS)
    bare = tokenizer.encode(prompt, add_special_tokens=False)
    decorated = tokenizer.encode(prompt, add_special_tokens=True)
    if bare != decorated:
        msg = (
            f"tokenizer adds {len(decorated) - len(bare)} special token(s) to a "
            "prompt; the eval harness would feed them and the loss mask here "
            "would not account for them"
        )
        raise ValueError(msg)


def percentile(values: list[int], q: float) -> int:
    """The q-th percentile by nearest rank. Empty input is a programming error."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def main(argv: list[str] | None = None) -> int:
    """Report the SFT dataset the protocol trains on.

    Prints the drop count and the sequence-length distribution: the first
    measured numbers of the block, and what Protocol section 7 needs beside
    the time per step.

        python -m verifiable_reward_lab.training.sft_data
    """
    from transformers import AutoTokenizer

    from verifiable_reward_lab.data.fetch import DATA_DIR
    from verifiable_reward_lab.data.gsm8k import load
    from verifiable_reward_lab.training.sft_config import SFTConfig

    parser = argparse.ArgumentParser(description="report the SFT training set")
    parser.add_argument("--config", type=Path, default=Path("configs/sft_qwen05b.yaml"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)

    try:
        config = SFTConfig.load(args.config)
        tokenizer = cast(
            "Tokenizer",
            AutoTokenizer.from_pretrained(
                config.model_name, revision=config.model_revision
            ),
        )
        check_no_special_tokens(tokenizer, "What is two plus two?")

        split = load(args.data_dir, seed=config.split_seed)
        dataset = build_dataset(split.train, tokenizer, max_seq_len=config.max_seq_len)
    except (ValueError, OSError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    pool = len(dataset) + dataset.n_dropped
    lengths = [len(e.input_ids) for e in dataset.examples]
    answers = [len(e.input_ids) - e.n_prompt_tokens for e in dataset.examples]
    steps_per_epoch = math.ceil(len(dataset) / config.effective_batch)

    print(f"model                  {config.model_name}")
    print(f"dev fingerprint        {split.fingerprint}")
    print(f"pool (train split)     {pool}  (split seed {config.split_seed})")
    print(f"kept                   {len(dataset)}")
    print(
        f"dropped > {config.max_seq_len:<4}         {dataset.n_dropped}"
        f"  ({dataset.n_dropped / pool:.2%})"
    )
    print(
        f"seq len p50/p95/max    {percentile(lengths, 50)}"
        f" / {percentile(lengths, 95)} / {max(lengths)}"
    )
    print(f"answer tokens p50/max  {percentile(answers, 50)} / {max(answers)}")
    print(
        f"steps                  {steps_per_epoch}/epoch x {config.epochs} epochs"
        f" = {steps_per_epoch * config.epochs}  (effective batch {config.effective_batch})"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
