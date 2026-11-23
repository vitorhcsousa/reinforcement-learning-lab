"""One run directory: created once, appended to, then closed with a summary.

    <out_dir>/
    ├── config.yaml     # the run that happened, written before the first step
    ├── metrics.jsonl   # one raw record per finished episode
    └── summary.json    # derived from metrics.jsonl on disk, written last

The directory is created exclusively: an existing one is refused, never
reused, so a rerun cannot overwrite or mix into earlier evidence. A run that
raises is still closed, with `status: failed` and the error, so a partial
directory never looks like a result.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import operator
import os
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from types import TracebackType

    from rl_lab.experiment.config import RunConfig

TRAIN_EPISODE = "train_episode"
EVAL_EPISODE = "eval_episode"

COMPLETE = "complete"
FAILED = "failed"

# summary keys that differ between two runs of the same config by design.
# everything else in summary.json, and all of metrics.jsonl, must not.
VOLATILE = ("wall_clock_s", "started_at", "finished_at")

_PACKAGES = ("numpy", "pyyaml", "torch", "verifiable-reward-lab")


class RunExistsError(FileExistsError):
    """The run directory already exists. Pick a new one; never reuse it."""


class BudgetExceededError(RuntimeError):
    """The run went over a resource limit in its config."""


def default_out_dir(config: RunConfig) -> Path:
    """config.out_dir if set, else runs/rl/<algorithm>/<env>/seed-<n>."""
    if config.out_dir is not None:
        return Path(config.out_dir)
    return Path("runs", "rl", config.algorithm_id, config.env_id, f"seed-{config.seed}")


class Run:
    """The evidence of one run, kept honest by construction.

    Use as a context manager; leaving the block closes the run, as complete
    on a clean exit and as failed on an exception, which is re-raised.
    """

    def __init__(self, out_dir: Path, config: RunConfig) -> None:
        """Create `out_dir` and write config.yaml.

        Raises:
            RunExistsError: `out_dir` already exists, even if empty.
        """
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_dir.mkdir()
        except FileExistsError:
            msg = (
                f"{out_dir} already exists; a run directory is never reused. "
                "Pass a new --out, or delete it if it is not evidence."
            )
            raise RunExistsError(msg) from None

        self.out_dir = out_dir
        self.config = config
        self.config_path = out_dir / "config.yaml"
        self.metrics_path = out_dir / "metrics.jsonl"
        self.summary_path = out_dir / "summary.json"

        config.save(self.config_path)
        self._metrics = self.metrics_path.open("x", encoding="utf-8")
        self._records = 0
        self._started = time.monotonic()
        self._started_at = _utc_now()
        self.summary: dict[str, Any] | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.summary is not None:
            return
        if exc is None:
            self.finish()
        else:
            self.finish(error=f"{type(exc).__name__}: {exc}")

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started

    def check_budget(self) -> None:
        """Raise if the run is over its wall-clock limit.

        Raises:
            BudgetExceededError: elapsed time exceeds max_wall_clock_s.
        """
        limit = self.config.max_wall_clock_s
        if self.elapsed_s > limit:
            msg = f"wall clock {self.elapsed_s:.1f}s exceeded max_wall_clock_s={limit}"
            raise BudgetExceededError(msg)

    def log_train_episode(
        self,
        *,
        episode: int,
        env_step: int,
        episode_return: float,
        length: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record one finished training episode.

        Args:
            episode: 0-based index among training episodes.
            env_step: Training environment steps taken when it finished.
            episode_return: Undiscounted sum of rewards, unrounded.
            length: Steps in the episode.
            terminated: Ended in a terminal state.
            truncated: Cut by the time limit without reaching one.
        """
        self._write(
            {
                "kind": TRAIN_EPISODE,
                "episode": _as_int("episode", episode),
                "env_step": _as_int("env_step", env_step),
                **_outcome(episode_return, length, terminated, truncated),
            }
        )

    def log_eval_episode(
        self,
        *,
        evaluation: int,
        env_step: int,
        episode: int,
        seed: int,
        episode_return: float,
        length: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record one evaluation episode.

        Args:
            evaluation: 0-based index of the evaluation it belongs to.
            env_step: Training environment steps taken when it was evaluated.
            episode: 0-based index within the evaluation.
            seed: The episode's reset seed, so it can be replayed alone.
            episode_return: Undiscounted sum of rewards, unrounded.
            length: Steps in the episode.
            terminated: Ended in a terminal state.
            truncated: Cut by the time limit without reaching one.
        """
        self._write(
            {
                "kind": EVAL_EPISODE,
                "evaluation": _as_int("evaluation", evaluation),
                "env_step": _as_int("env_step", env_step),
                "episode": _as_int("episode", episode),
                "seed": _as_int("seed", seed),
                **_outcome(episode_return, length, terminated, truncated),
            }
        )

    def finish(self, *, error: str | None = None) -> dict[str, Any]:
        """Close the run and write summary.json, derived from metrics.jsonl.

        A clean finish also checks the run against its config: every
        evaluation has eval_episodes episodes. A run that fails the check is
        closed as failed and the check's error is raised.

        Args:
            error: Why the run failed. None closes it as complete.

        Returns:
            The summary that was written.

        Raises:
            RuntimeError: The run was already closed.
            ValueError: A clean finish found the metrics inconsistent.
        """
        if self.summary is not None:
            msg = f"{self.out_dir} is already closed"
            raise RuntimeError(msg)
        self._metrics.close()

        records = [
            json.loads(line)
            for line in self.metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        train = [r for r in records if r["kind"] == TRAIN_EPISODE]
        evals = [r for r in records if r["kind"] == EVAL_EPISODE]
        evaluations = _evaluations(evals)

        problem = None if error is not None else self._inconsistency(evaluations)
        if problem is not None:
            error = f"ValueError: {problem}"

        self.summary = self._write_summary(error, train, evaluations, evals)
        if problem is not None:
            raise ValueError(problem)
        return self.summary

    def _inconsistency(self, evaluations: list[dict[str, Any]]) -> str | None:
        """Why a run that raised nothing still cannot be called complete."""
        if not evaluations:
            return "the run recorded no evaluation"
        for e in evaluations:
            if e["episodes"] != self.config.eval_episodes:
                return (
                    f"evaluation {e['evaluation']} has {e['episodes']} episodes, "
                    f"config says {self.config.eval_episodes}"
                )
        return None

    def _write(self, record: dict[str, Any]) -> None:
        if self.summary is not None:
            msg = f"{self.out_dir} is closed; nothing more can be recorded"
            raise RuntimeError(msg)
        # json writes floats with repr, which round-trips exactly: the file
        # holds the number that was measured, not a printed approximation
        self._metrics.write(json.dumps(record, allow_nan=False) + "\n")
        self._metrics.flush()
        self._records += 1

    def _write_summary(
        self,
        error: str | None,
        train: list[dict[str, Any]],
        evaluations: list[dict[str, Any]],
        evals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        final = evaluations[-1] if evaluations else None
        final_returns = (
            [r["return"] for r in evals if r["evaluation"] == final["evaluation"]]
            if final is not None
            else []
        )
        summary: dict[str, Any] = {
            "status": COMPLETE if error is None else FAILED,
            "error": error,
            "env_id": self.config.env_id,
            "algorithm_id": self.config.algorithm_id,
            "seed": self.config.seed,
            "config_sha256": self.config.sha256(),
            "config": self.config.to_dict(),
            "files": {
                "config": {
                    "path": self.config_path.name,
                    "sha256": _sha256(self.config_path),
                },
                "metrics": {
                    "path": self.metrics_path.name,
                    "sha256": _sha256(self.metrics_path),
                    "records": self._records,
                },
            },
            "train": {
                "episodes": len(train),
                "env_steps_at_last_episode": train[-1]["env_step"] if train else 0,
                "mean_return": _mean([r["return"] for r in train]),
                "terminated": sum(r["terminated"] for r in train),
                "truncated": sum(r["truncated"] for r in train),
            },
            "eval": {
                "episodes_per_evaluation": self.config.eval_episodes,
                "evaluations": evaluations,
                "final": None
                if final is None
                else {
                    "evaluation": final["evaluation"],
                    "env_step": final["env_step"],
                    "mean_return": final["mean_return"],
                    "returns": final_returns,
                },
            },
            "wall_clock_s": self.elapsed_s,
            "started_at": self._started_at,
            "finished_at": _utc_now(),
            "versions": {
                "python": platform.python_version(),
                **{name: _version(name) for name in _PACKAGES},
            },
            "platform": platform.platform(),
            "git": _git_state(),
        }
        # written aside and renamed, so a crash mid-write leaves no
        # half-summary that a reader could mistake for a whole one
        tmp = self.summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", "utf-8")
        os.replace(tmp, self.summary_path)
        return summary


def _outcome(
    episode_return: float, length: int, terminated: bool, truncated: bool
) -> dict[str, Any]:
    value = _as_float("episode_return", episode_return)
    steps = _as_int("length", length)
    if steps < 1:
        msg = f"length must be >= 1, got {steps}"
        raise ValueError(msg)
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        msg = (
            f"terminated and truncated must be bools, got {terminated!r}, {truncated!r}"
        )
        raise ValueError(msg)
    if not (terminated or truncated):
        # only finished episodes are recorded; a return cut short by the
        # training budget is not a return
        msg = "an episode record must be terminated or truncated"
        raise ValueError(msg)
    return {
        "return": value,
        "length": steps,
        "terminated": terminated,
        "truncated": truncated,
    }


def _evaluations(evals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One aggregate per evaluation, in the order they were run."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in evals:
        grouped.setdefault(record["evaluation"], []).append(record)
    return [
        {
            "evaluation": index,
            "env_step": group[0]["env_step"],
            "episodes": len(group),
            "mean_return": _mean([r["return"] for r in group]),
            "terminated": sum(r["terminated"] for r in group),
            "truncated": sum(r["truncated"] for r in group),
        }
        for index, group in grouped.items()
    ]


def _as_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        msg = f"{name} must be an integer, got {value!r}"
        raise ValueError(msg)
    try:
        return operator.index(value)  # ty: ignore[invalid-argument-type]
    except TypeError:
        msg = f"{name} must be an integer, got {value!r}"
        raise ValueError(msg) from None


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} must be a number, got {value!r}"
        raise ValueError(msg)
    if not math.isfinite(value):
        msg = f"{name} must be finite, got {value}"
        raise ValueError(msg)
    return float(value)


def _mean(values: list[float]) -> float | None:
    # fsum: the exactly rounded sum, independent of the order of accumulation
    return math.fsum(values) / len(values) if values else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_state() -> dict[str, Any]:
    """The commit the code ran from, and whether the tree had local changes.

    Both None outside a git checkout, e.g. from an installed wheel.
    """
    here = str(Path(__file__).resolve().parent)

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", here, *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = git("status", "--porcelain")
    return {
        "commit": commit.strip(),
        "dirty": None if status is None else bool(status.strip()),
    }
