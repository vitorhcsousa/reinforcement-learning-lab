"""The SFT data path: prompt agreement, the loss mask, drops and padding."""

from __future__ import annotations

import re

import pytest
import torch

from verifiable_reward_lab.data.gsm8k import Example
from verifiable_reward_lab.evaluation.prompt import build_prompt
from verifiable_reward_lab.training.sft_data import (
    IGNORE_INDEX,
    N_SHOTS,
    SFTExample,
    build_dataset,
    check_no_special_tokens,
    collate,
    encode_example,
    percentile,
    strip_annotations,
)

QUESTION = "A baker has 12 trays with 8 muffins each. How many muffins are there?"
RAW_ANSWER = "The trays hold 12 * 8 = <<12*8=96>>96 muffins.\n#### 96"
CLEAN_ANSWER = "The trays hold 12 * 8 = 96 muffins.\n#### 96"

PIECE = re.compile(r"\s*\S+|\s+")


class FakeTokenizer:
    """A tokenizer that merges a leading space into the token after it.

    Every real BPE vocabulary does this, and it is the one property the
    module's correctness depends on. A fake that tokenised character by
    character would happily pass an implementation that derives the mask
    boundary by arithmetic over a single joint encoding.

    Ids are handed out in first-seen order starting at 1, so eos (0) can
    never collide with a piece.
    """

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self.eos_token_id: int | None = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [
            self._ids.setdefault(p, len(self._ids) + 1) for p in PIECE.findall(text)
        ]


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def example() -> Example:
    return Example(question=QUESTION, answer=RAW_ANSWER)


def test_prompt_ids_are_the_eval_harness_ids(
    tokenizer: FakeTokenizer, example: Example
) -> None:
    """The masked prefix is what the eval harness feeds, token for token."""
    encoded = encode_example(example, tokenizer)
    expected = tokenizer.encode(build_prompt(QUESTION, N_SHOTS))

    assert list(encoded.input_ids[: encoded.n_prompt_tokens]) == expected


def test_a_joint_encoding_would_have_moved_the_boundary(
    tokenizer: FakeTokenizer, example: Example
) -> None:
    """Why the two halves are encoded separately.

    Not a property of the module so much as the reason it exists: encoding
    prompt and answer in one call puts different ids at the boundary, so a
    mask placed by counting the prompt's tokens would cover different text.
    """
    encoded = encode_example(example, tokenizer)
    joint = tokenizer.encode(build_prompt(QUESTION, N_SHOTS) + CLEAN_ANSWER)

    assert joint[: encoded.n_prompt_tokens] != list(
        encoded.input_ids[: encoded.n_prompt_tokens]
    )


def test_labels_mask_the_prompt_and_nothing_else(
    tokenizer: FakeTokenizer, example: Example
) -> None:
    n = encode_example(example, tokenizer)
    labels = n.labels

    assert all(label == IGNORE_INDEX for label in labels[: n.n_prompt_tokens])
    assert all(label != IGNORE_INDEX for label in labels[n.n_prompt_tokens :])
    assert list(labels[n.n_prompt_tokens :]) == list(n.input_ids[n.n_prompt_tokens :])


def test_eos_is_learned_not_masked(tokenizer: FakeTokenizer, example: Example) -> None:
    """Mask the eos and the model never stops; greedy decoding then runs to
    the token cap on every single evaluation example."""
    encoded = encode_example(example, tokenizer)

    assert encoded.input_ids[-1] == tokenizer.eos_token_id
    assert encoded.labels[-1] == tokenizer.eos_token_id


def test_answer_loses_its_calculator_annotations(
    tokenizer: FakeTokenizer, example: Example
) -> None:
    assert strip_annotations(RAW_ANSWER) == CLEAN_ANSWER

    encoded = encode_example(example, tokenizer)
    answer_ids = list(encoded.input_ids[encoded.n_prompt_tokens :])

    assert answer_ids == [*tokenizer.encode(CLEAN_ANSWER), 0]


def test_a_tokenizer_without_eos_is_rejected(example: Example) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = None

    with pytest.raises(ValueError, match="eos_token_id"):
        encode_example(example, tokenizer)


def test_long_examples_are_dropped_and_counted(tokenizer: FakeTokenizer) -> None:
    short = Example(question="Two plus two?", answer="#### 4")
    long = Example(question=QUESTION, answer=RAW_ANSWER)

    encoded_short = encode_example(short, tokenizer)
    cap = len(encoded_short.input_ids)

    dataset = build_dataset([short, long, short], tokenizer, max_seq_len=cap)

    assert len(dataset) == 2
    assert dataset.n_dropped == 1
    assert dataset.max_seq_len == cap


def test_the_drop_count_is_deterministic(tokenizer: FakeTokenizer) -> None:
    """The number goes in the protocol, so it has to be the same number."""
    examples = [
        Example(question=QUESTION, answer=RAW_ANSWER),
        Example(question="Two plus two?", answer="#### 4"),
    ]

    first = build_dataset(examples, tokenizer, max_seq_len=12)
    second = build_dataset(examples, FakeTokenizer(), max_seq_len=12)

    assert first.n_dropped == second.n_dropped
    assert len(first) == len(second)


def test_build_dataset_keeps_split_order(tokenizer: FakeTokenizer) -> None:
    examples = [
        Example(question="Two plus two?", answer="#### 4"),
        Example(question="Three plus three?", answer="#### 6"),
    ]

    dataset = build_dataset(examples, tokenizer, max_seq_len=64)
    expected = [encode_example(e, tokenizer) for e in examples]

    assert list(dataset.examples) == expected


def test_build_dataset_rejects_a_nonsense_cap(tokenizer: FakeTokenizer) -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        build_dataset([], tokenizer, max_seq_len=0)


def test_collate_pads_right_and_ignores_the_padding() -> None:
    short = SFTExample(input_ids=(5, 6), labels=(IGNORE_INDEX, 6), n_prompt_tokens=1)
    long = SFTExample(
        input_ids=(5, 6, 7, 8), labels=(IGNORE_INDEX, 6, 7, 8), n_prompt_tokens=1
    )

    batch = collate([short, long], pad_id=99)

    assert batch["input_ids"].shape == (2, 4)
    assert torch.equal(batch["input_ids"][0], torch.tensor([5, 6, 99, 99]))
    assert torch.equal(
        batch["labels"][0], torch.tensor([IGNORE_INDEX, 6, IGNORE_INDEX, IGNORE_INDEX])
    )
    assert torch.equal(batch["attention_mask"][0], torch.tensor([1, 1, 0, 0]))
    assert torch.equal(batch["attention_mask"][1], torch.tensor([1, 1, 1, 1]))


def test_collate_returns_long_tensors() -> None:
    example = SFTExample(input_ids=(5, 6), labels=(IGNORE_INDEX, 6), n_prompt_tokens=1)

    batch = collate([example], pad_id=0)

    assert all(t.dtype == torch.long for t in batch.values())


def test_collate_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate([], pad_id=0)


class BosTokenizer(FakeTokenizer):
    """A tokenizer that prepends a bos when asked for special tokens."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = super().encode(text)
        return [-1, *ids] if add_special_tokens else ids


def test_a_tokenizer_that_adds_nothing_passes(tokenizer: FakeTokenizer) -> None:
    check_no_special_tokens(tokenizer, QUESTION)


def test_a_tokenizer_that_adds_a_bos_is_caught() -> None:
    """The eval harness takes the default, this file does not. If they ever
    disagree the mask is off by one and nothing else complains."""
    with pytest.raises(ValueError, match="special token"):
        check_no_special_tokens(BosTokenizer(), QUESTION)


def test_percentile_is_nearest_rank() -> None:
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile([7], 50) == 7


def test_bucketing_rounds_the_width_up() -> None:
    """Five possible shapes instead of one per batch."""
    short = SFTExample(input_ids=(5, 6), labels=(IGNORE_INDEX, 6), n_prompt_tokens=1)
    long = SFTExample(
        input_ids=(5, 6, 7, 8, 9), labels=(IGNORE_INDEX, 6, 7, 8, 9), n_prompt_tokens=1
    )

    batch = collate([short, long], pad_id=99, pad_to_multiple_of=4)

    assert batch["input_ids"].shape == (2, 8)


def test_bucketing_does_not_change_what_the_loss_sees() -> None:
    """The added positions are masked twice over, so rounding the width up
    is a compute cost and nothing else."""
    example = SFTExample(
        input_ids=(5, 6, 7), labels=(IGNORE_INDEX, 6, 7), n_prompt_tokens=1
    )

    tight = collate([example], pad_id=99)
    bucketed = collate([example], pad_id=99, pad_to_multiple_of=8)

    assert bucketed["input_ids"].shape == (1, 8)
    assert torch.equal(bucketed["labels"][0, :3], tight["labels"][0])
    assert torch.all(bucketed["labels"][0, 3:] == IGNORE_INDEX)
    assert torch.all(bucketed["attention_mask"][0, 3:] == 0)


def test_an_exact_multiple_is_left_alone() -> None:
    example = SFTExample(
        input_ids=(5, 6, 7, 8), labels=(IGNORE_INDEX, 6, 7, 8), n_prompt_tokens=1
    )

    batch = collate([example], pad_id=0, pad_to_multiple_of=4)

    assert batch["input_ids"].shape == (1, 4)


def test_a_nonsense_bucket_is_rejected() -> None:
    example = SFTExample(input_ids=(5, 6), labels=(IGNORE_INDEX, 6), n_prompt_tokens=1)

    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        collate([example], pad_id=0, pad_to_multiple_of=0)


def test_the_protocol_cap_is_a_whole_number_of_buckets() -> None:
    """640 is five buckets of 128. A cap that was not would give a sixth
    shape for the longest batches alone."""
    assert 640 % 128 == 0
