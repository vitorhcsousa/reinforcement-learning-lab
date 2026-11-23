"""One seed, independent streams: evaluation never depends on what training drew."""

from __future__ import annotations

import random
import sys

import numpy as np
import pytest

from rl_lab.experiment.seeding import (
    MAX_SEED,
    episode_policy_rng,
    make_streams,
    seed_libraries,
)


def test_the_same_seed_gives_the_same_streams() -> None:
    a, b = make_streams(3, eval_episodes=8), make_streams(3, eval_episodes=8)

    assert a.train_env_seed == b.train_env_seed
    assert a.eval_episode_seeds == b.eval_episode_seeds
    assert np.array_equal(a.policy.random(16), b.policy.random(16))


def test_different_seeds_give_different_streams() -> None:
    a, b = make_streams(3, eval_episodes=8), make_streams(4, eval_episodes=8)

    assert a.train_env_seed != b.train_env_seed
    assert a.eval_episode_seeds != b.eval_episode_seeds
    assert not np.array_equal(a.policy.random(16), b.policy.random(16))


def test_the_streams_of_one_run_are_distinct() -> None:
    """Spawned, not offset: no stream is another one shifted by a step."""
    streams = make_streams(0, eval_episodes=16)
    eval_seeds = set(streams.eval_episode_seeds)

    assert len(eval_seeds) == 16
    assert streams.train_env_seed not in eval_seeds


def test_evaluation_seeds_do_not_depend_on_training_draws() -> None:
    """The property a fair comparison rests on: two algorithms that consume
    the policy stream differently are still evaluated on the same episodes."""
    quiet = make_streams(5, eval_episodes=4)
    busy = make_streams(5, eval_episodes=4)
    busy.policy.random(10_000)

    assert busy.eval_episode_seeds == quiet.eval_episode_seeds


def test_the_episode_policy_stream_is_not_the_environment_stream() -> None:
    seed = make_streams(0, eval_episodes=1).eval_episode_seeds[0]
    env = np.random.default_rng(seed)  # what RandomWalk.reset(seed=seed) uses
    policy = episode_policy_rng(seed)

    assert not np.array_equal(env.random(16), policy.random(16))
    assert np.array_equal(
        episode_policy_rng(seed).random(16), episode_policy_rng(seed).random(16)
    )


def test_numpy_global_state_is_left_alone() -> None:
    """Code that reaches for np.random.* should fail the same-seed test, not
    pass it by accident."""
    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)

    make_streams(9, eval_episodes=4)
    seed_libraries(9)

    assert np.array_equal(np.random.random(4), expected)


def test_python_random_is_seeded() -> None:
    seed_libraries(11)
    first = random.random()
    seed_libraries(11)
    assert random.random() == first


def test_torch_is_seeded_only_when_asked() -> None:
    import torch

    seed_libraries(11, torch=True)
    first = torch.rand(3)
    seed_libraries(11, torch=True)
    assert torch.equal(torch.rand(3), first)


def test_torch_is_not_imported_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    # hide any torch the rest of the suite imported, then check it stays hidden
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    seed_libraries(11)
    assert "torch" not in sys.modules


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1, 1.0, True, "1"])
def test_unusable_seeds_are_refused(seed: object) -> None:
    with pytest.raises(ValueError, match="seed must be"):
        make_streams(seed, eval_episodes=1)  # ty: ignore[invalid-argument-type]


def test_the_extreme_seeds_are_usable() -> None:
    make_streams(0, eval_episodes=1)
    make_streams(MAX_SEED, eval_episodes=1)
