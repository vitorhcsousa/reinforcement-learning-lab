"""One seed in, independent random streams out: environment, policy, evaluation.

Randomness is passed around as explicit `numpy.random.Generator`s derived
with `SeedSequence.spawn`, never read from numpy's global state. The global
state is deliberately left unseeded: code that reaches for it then fails the
same-seed test instead of being reproducible by accident.

Determinism is promised on one machine with one set of library versions.
NumPy keeps a bit generator's stream stable across releases, but not every
distribution method built on it (NEP 19); torch kernels differ across
devices and builds. Cross-platform agreement is something to measure, not
assume.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

# the widest seed every consumer accepts: python, numpy and torch all take
# a uint32, and some legacy paths take nothing wider
MAX_SEED = 2**32 - 1


def check_seed(seed: object) -> int:
    """Return `seed` if it is a usable seed, or say why it is not.

    Raises:
        ValueError: Not an int (bools included), or outside [0, 2**32 - 1].
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        msg = f"seed must be an integer, got {seed!r}"
        raise ValueError(msg)
    if not 0 <= seed <= MAX_SEED:
        msg = f"seed must be in [0, {MAX_SEED}], got {seed}"
        raise ValueError(msg)
    return seed


@dataclass(frozen=True)
class Streams:
    """The randomness of one run, split by who consumes it.

    The split is the point. Training draws from `train_env` and `policy`;
    evaluation draws only from `eval_episode_seeds`. So the evaluation
    episodes are the same whatever training did, and two algorithms run with
    the same seed are evaluated on the same episodes - a paired comparison.
    """

    train_env_seed: int
    """Seeds the training environment's first reset; later resets continue it."""

    policy: np.random.Generator
    """Action sampling during training."""

    eval_episode_seeds: tuple[int, ...]
    """One per evaluation episode, reused by every evaluation of the run."""


def make_streams(seed: int, *, eval_episodes: int) -> Streams:
    """Derive a run's streams from its seed.

    Args:
        seed: The run's seed, from RunConfig.
        eval_episodes: How many evaluation episode seeds to derive.

    Returns:
        Independent streams; the same seed always gives the same ones.
    """
    check_seed(seed)
    train_env, policy, evaluation = np.random.SeedSequence(seed).spawn(3)
    return Streams(
        train_env_seed=int(train_env.generate_state(1)[0]),
        policy=np.random.default_rng(policy),
        eval_episode_seeds=tuple(
            int(s) for s in evaluation.generate_state(eval_episodes)
        ),
    )


def episode_policy_rng(episode_seed: int) -> np.random.Generator:
    """The policy's stream for one evaluation episode.

    A child of the episode seed, so it is independent of the environment
    stream that `reset(seed=episode_seed)` starts from.
    """
    (child,) = np.random.SeedSequence(episode_seed).spawn(1)
    return np.random.default_rng(child)


def seed_libraries(seed: int, *, torch: bool = False) -> None:
    """Seed the global state of libraries that do not take a generator.

    Python's `random` always; torch only when the run uses it, so a run that
    does not is not made to import it.

    Args:
        seed: The run's seed.
        torch: Also call torch.manual_seed.
    """
    check_seed(seed)
    random.seed(seed)
    if torch:
        import torch as _torch

        _torch.manual_seed(seed)
