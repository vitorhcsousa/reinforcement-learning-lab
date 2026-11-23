"""A tiny CPU experiment with a known answer, to exercise the run contract end to end.

    uv run python -m rl_lab.experiment.smoke --config configs/rl_smoke.yaml

A one-dimensional random walk on positions 0..length, starting at `start`.
Each step the agent pushes left or right; with probability `slip` the push
is reversed. Reaching `length` pays +1 and terminates, reaching 0 pays 0 and
terminates, and `max_episode_steps` truncates. The policy is scripted - it
moves right with a fixed probability - so nothing learns: training and
evaluation differ only in that probability. With no time limit the expected
return is the gambler's-ruin probability (`reach_probability`), which is what
the tests check the measured mean against.

Not an environment registry and not an algorithm. Those are later stages.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rl_lab.experiment.config import RunConfig
from rl_lab.experiment.run import Run, RunExistsError, default_out_dir
from rl_lab.experiment.seeding import episode_policy_rng, make_streams, seed_libraries

if TYPE_CHECKING:
    from collections.abc import Sequence

ENV_ID = "RandomWalk-v0"
ALGORITHM_ID = "scripted_walk"

LEFT, RIGHT = -1, 1


class RandomWalk:
    """The walk, with gymnasium's reset/step semantics and no gymnasium."""

    def __init__(
        self, *, length: int, start: int, slip: float, max_episode_steps: int
    ) -> None:
        if length < 2:
            msg = f"length must be >= 2, got {length}"
            raise ValueError(msg)
        if not 0 < start < length:
            msg = f"start must be strictly between 0 and length={length}, got {start}"
            raise ValueError(msg)
        if not 0.0 <= slip <= 1.0:
            msg = f"slip must be a probability, got {slip}"
            raise ValueError(msg)
        self.length = length
        self.start = start
        self.slip = slip
        self.max_episode_steps = max_episode_steps
        self._rng: np.random.Generator | None = None
        self._position = start
        self._t = 0

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        """Start an episode. A seed restarts the stream; None continues it."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            msg = "the first reset needs a seed"
            raise RuntimeError(msg)
        self._position = self.start
        self._t = 0
        return self._position, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Push left or right. Returns obs, reward, terminated, truncated, info."""
        if action not in (LEFT, RIGHT):
            msg = f"action must be {LEFT} or {RIGHT}, got {action}"
            raise ValueError(msg)
        if self._rng is None:
            msg = "step before reset"
            raise RuntimeError(msg)
        slipped = self._rng.random() < self.slip
        self._position += -action if slipped else action
        self._t += 1

        terminated = self._position in (0, self.length)
        reward = 1.0 if self._position == self.length else 0.0
        # terminated wins: reaching a terminal state on the last allowed step
        # is a true return, not a cut-off one
        truncated = not terminated and self._t >= self.max_episode_steps
        return self._position, reward, terminated, truncated, {}


def act(p_right: float, rng: np.random.Generator) -> int:
    """The scripted policy: right with probability p_right."""
    return RIGHT if rng.random() < p_right else LEFT


def reach_probability(p_right: float, slip: float, *, length: int, start: int) -> float:
    """Expected return with no time limit: P(reach `length` before 0).

    The walk moves right with q = p(1 - slip) + (1 - p) slip, and from
    `start` reaches `length` first with (1 - r^start) / (1 - r^length),
    r = (1 - q) / q - or start / length when q = 1/2.
    """
    q = p_right * (1.0 - slip) + (1.0 - p_right) * slip
    if q == 0.0:
        return 0.0
    if q == 0.5:
        return start / length
    r = (1.0 - q) / q
    return (1.0 - r**start) / (1.0 - r**length)


@dataclasses.dataclass(frozen=True)
class _Params:
    length: int
    start: int
    slip: float
    train_p_right: float
    eval_p_right: float

    @classmethod
    def from_config(cls, config: RunConfig) -> _Params:
        if config.env_id != ENV_ID or config.algorithm_id != ALGORITHM_ID:
            msg = (
                f"the smoke experiment runs {ALGORITHM_ID} on {ENV_ID}, "
                f"got {config.algorithm_id} on {config.env_id}"
            )
            raise ValueError(msg)
        env, alg = config.env_params, config.algorithm_params
        expected_env, expected_alg = (
            {"length", "start", "slip"},
            {
                "train_p_right",
                "eval_p_right",
            },
        )
        if set(env) != expected_env:
            msg = (
                f"env_params must be exactly {sorted(expected_env)}, got {sorted(env)}"
            )
            raise ValueError(msg)
        if set(alg) != expected_alg:
            msg = (
                f"algorithm_params must be exactly {sorted(expected_alg)}, "
                f"got {sorted(alg)}"
            )
            raise ValueError(msg)
        for name in expected_alg:
            p = alg[name]
            if isinstance(p, bool | str) or not 0.0 <= p <= 1.0:
                msg = f"algorithm_params.{name} must be a probability, got {p!r}"
                raise ValueError(msg)
        length, start, slip = env["length"], env["start"], env["slip"]
        if not isinstance(length, int) or not isinstance(start, int):
            msg = f"length and start must be integers, got {length!r}, {start!r}"
            raise ValueError(msg)
        if isinstance(slip, bool | str):
            msg = f"slip must be a probability, got {slip!r}"
            raise ValueError(msg)
        return cls(
            length=length,
            start=start,
            slip=float(slip),
            train_p_right=float(alg["train_p_right"]),
            eval_p_right=float(alg["eval_p_right"]),
        )


def run_smoke(config: RunConfig, out_dir: Path) -> dict[str, Any]:
    """Train (scripted), evaluate, and write the run to `out_dir`.

    Args:
        config: A RandomWalk-v0 / scripted_walk config.
        out_dir: Must not exist yet.

    Returns:
        The run's summary.

    Raises:
        ValueError: The config is not for this experiment.
        RunExistsError: out_dir already exists.
        BudgetExceededError: The run went over max_wall_clock_s.
    """
    params = _Params.from_config(config)
    streams = make_streams(config.seed, eval_episodes=config.eval_episodes)
    seed_libraries(config.seed)

    def make_env() -> RandomWalk:
        return RandomWalk(
            length=params.length,
            start=params.start,
            slip=params.slip,
            max_episode_steps=config.max_episode_steps,
        )

    # a separate instance, so evaluation never advances the training stream
    train_env, eval_env = make_env(), make_env()
    evaluation = 0

    with Run(out_dir, config) as run:

        def evaluate(env_step: int) -> None:
            nonlocal evaluation
            for episode, seed in enumerate(streams.eval_episode_seeds):
                run.check_budget()
                eval_env.reset(seed=seed)
                rng = episode_policy_rng(seed)
                total, length = 0.0, 0
                while True:
                    _, reward, terminated, truncated, _ = eval_env.step(
                        act(params.eval_p_right, rng)
                    )
                    total += reward
                    length += 1
                    if terminated or truncated:
                        break
                run.log_eval_episode(
                    evaluation=evaluation,
                    env_step=env_step,
                    episode=episode,
                    seed=seed,
                    episode_return=total,
                    length=length,
                    terminated=terminated,
                    truncated=truncated,
                )
            evaluation += 1

        evaluate(0)

        train_env.reset(seed=streams.train_env_seed)
        episode, total, length = 0, 0.0, 0
        for step in range(1, config.train_steps + 1):
            run.check_budget()
            _, reward, terminated, truncated, _ = train_env.step(
                act(params.train_p_right, streams.policy)
            )
            total += reward
            length += 1
            if terminated or truncated:
                run.log_train_episode(
                    episode=episode,
                    env_step=step,
                    episode_return=total,
                    length=length,
                    terminated=terminated,
                    truncated=truncated,
                )
                episode, total, length = episode + 1, 0.0, 0
                train_env.reset()
            if step % config.eval_interval == 0:
                evaluate(step)
        # the episode in progress when the budget ran out is dropped: it has
        # no return yet, and recording a partial one would bias the mean down
        if config.train_steps % config.eval_interval != 0:
            evaluate(config.train_steps)

        return run.finish()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. 0 on a complete run, 1 if it failed, 2 if it never started."""
    parser = argparse.ArgumentParser(description="run the RL smoke experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/rl_smoke.yaml"))
    parser.add_argument(
        "--out", type=Path, default=None, help="run directory; must not exist"
    )
    parser.add_argument("--seed", type=int, default=None, help="override the seed")
    args = parser.parse_args(argv)

    try:
        config = RunConfig.load(args.config)
        if args.seed is not None:
            config = dataclasses.replace(config, seed=args.seed)
        out_dir = args.out if args.out is not None else default_out_dir(config)
        # the saved config says where it was written, overrides included
        config = dataclasses.replace(config, out_dir=str(out_dir))
        _Params.from_config(config)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        summary = run_smoke(config, out_dir)
    except RunExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - the summary already records it
        print(f"run failed: {e} (see {out_dir / 'summary.json'})", file=sys.stderr)
        return 1

    train, final = summary["train"], summary["eval"]["final"]
    print(
        f"train: {train['episodes']} episodes, mean return {train['mean_return']} | "
        f"eval @ {final['env_step']}: mean return {final['mean_return']} "
        f"over {len(final['returns'])} episodes | {out_dir}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
