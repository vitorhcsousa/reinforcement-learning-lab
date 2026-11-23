"""The run directory: raw numbers in, a summary that points at them out, nothing overwritten."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rl_lab.experiment.config import RunConfig
from rl_lab.experiment.run import (
    COMPLETE,
    EVAL_EPISODE,
    FAILED,
    TRAIN_EPISODE,
    BudgetExceededError,
    Run,
    RunExistsError,
    default_out_dir,
)


def config(**overrides: Any) -> RunConfig:
    base: dict[str, Any] = {
        "env_id": "Test-v0",
        "algorithm_id": "test",
        "seed": 0,
        "train_steps": 10,
        "eval_interval": 10,
        "eval_episodes": 2,
        "max_episode_steps": 5,
    }
    base.update(overrides)
    return RunConfig(**base)


def outcome(episode_return: float = 1.0, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "episode_return": episode_return,
        "length": 3,
        "terminated": True,
        "truncated": False,
    }
    base.update(overrides)
    return base


def log_evaluation(run: Run, index: int, env_step: int, returns: list[float]) -> None:
    for episode, value in enumerate(returns):
        run.log_eval_episode(
            evaluation=index,
            env_step=env_step,
            episode=episode,
            seed=100 + episode,
            **outcome(value),
        )


def records(run_dir: Path) -> list[dict[str, Any]]:
    text = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def test_a_run_directory_holds_config_metrics_and_summary(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with Run(out, config()) as run:
        log_evaluation(run, 0, 0, [1.0, 0.0])

    assert sorted(p.name for p in out.iterdir()) == [
        "config.yaml",
        "metrics.jsonl",
        "summary.json",
    ]
    assert RunConfig.load(out / "config.yaml") == config()
    assert summary(out)["status"] == COMPLETE


@pytest.mark.parametrize(
    "value", [0.1 + 0.2, 1 / 3, 1e-300, 123456789.123456789, -2.5e-17]
)
def test_raw_values_survive_exactly(tmp_path: Path, value: float) -> None:
    """No rounding anywhere between the measurement and the file."""
    out = tmp_path / "run"
    with Run(out, config(eval_episodes=1)) as run:
        run.log_train_episode(episode=0, env_step=3, **outcome(value))
        log_evaluation(run, 0, 0, [value])

    train, evaluation = records(out)
    assert train["return"] == value
    assert evaluation["return"] == value
    assert summary(out)["eval"]["final"]["returns"] == [value]


def test_numpy_scalars_are_recorded_as_plain_numbers(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with Run(out, config(eval_episodes=1)) as run:
        run.log_train_episode(
            episode=np.int64(0),  # ty: ignore[invalid-argument-type]
            env_step=np.int64(3),  # ty: ignore[invalid-argument-type]
            **outcome(np.float64(0.1) + np.float64(0.2)),
        )
        log_evaluation(run, 0, 0, [1.0])

    assert records(out)[0]["return"] == 0.1 + 0.2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "1.0", True])
def test_a_return_that_is_not_a_finite_number_is_refused(
    tmp_path: Path, value: object
) -> None:
    with Run(tmp_path / "run", config()) as run:
        with pytest.raises(ValueError, match="episode_return must be"):
            run.log_train_episode(episode=0, env_step=1, **outcome(value))  # ty: ignore[invalid-argument-type]
        log_evaluation(run, 0, 0, [1.0, 1.0])


def test_an_unfinished_episode_cannot_be_recorded(tmp_path: Path) -> None:
    with Run(tmp_path / "run", config()) as run:
        with pytest.raises(ValueError, match="terminated or truncated"):
            run.log_train_episode(
                episode=0, env_step=1, **outcome(terminated=False, truncated=False)
            )
        log_evaluation(run, 0, 0, [1.0, 1.0])


def test_training_and_evaluation_are_recorded_separately(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with Run(out, config()) as run:
        log_evaluation(run, 0, 0, [0.0, 0.0])
        run.log_train_episode(episode=0, env_step=4, **outcome(1.0))
        run.log_train_episode(
            episode=1, env_step=9, **outcome(0.5, terminated=False, truncated=True)
        )
        log_evaluation(run, 1, 10, [1.0, 0.0])

    kinds = [r["kind"] for r in records(out)]
    assert kinds.count(TRAIN_EPISODE) == 2
    assert kinds.count(EVAL_EPISODE) == 4

    s = summary(out)
    assert s["train"]["episodes"] == 2
    assert s["train"]["mean_return"] == 0.75
    assert (s["train"]["terminated"], s["train"]["truncated"]) == (1, 1)
    assert [e["mean_return"] for e in s["eval"]["evaluations"]] == [0.0, 0.5]
    assert [e["env_step"] for e in s["eval"]["evaluations"]] == [0, 10]
    assert s["eval"]["final"]["returns"] == [1.0, 0.0]


def test_the_summary_points_at_its_exact_config_and_metrics(tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg = config(seed=42)
    with Run(out, cfg) as run:
        log_evaluation(run, 0, 0, [1.0, 0.0])

    s = summary(out)
    files = s["files"]
    assert files["config"]["sha256"] == _sha256(out / files["config"]["path"])
    assert files["metrics"]["sha256"] == _sha256(out / files["metrics"]["path"])
    assert files["metrics"]["records"] == len(records(out))
    assert s["config_sha256"] == RunConfig.load(out / "config.yaml").sha256()
    assert RunConfig.from_dict(s["config"]) == cfg
    assert (s["seed"], s["env_id"], s["algorithm_id"]) == (42, "Test-v0", "test")
    assert s["eval"]["episodes_per_evaluation"] == 2


def test_the_summary_records_versions_and_the_commit(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with Run(out, config()) as run:
        log_evaluation(run, 0, 0, [1.0, 1.0])

    s = summary(out)
    assert s["versions"]["python"]
    assert s["versions"]["numpy"] == np.__version__
    # this suite runs from a git checkout, so the commit is known
    assert len(s["git"]["commit"]) == 40
    assert isinstance(s["git"]["dirty"], bool)


def test_an_existing_run_directory_is_refused_and_left_untouched(
    tmp_path: Path,
) -> None:
    out = tmp_path / "run"
    with Run(out, config()) as run:
        log_evaluation(run, 0, 0, [1.0, 1.0])
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    with pytest.raises(RunExistsError, match="never reused"):
        Run(out, config(seed=1))

    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_even_an_empty_existing_directory_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(RunExistsError):
        Run(out, config())


def test_a_run_that_raises_is_closed_as_failed(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="diverged"), Run(out, config()) as run:
        log_evaluation(run, 0, 0, [1.0, 1.0])
        msg = "diverged"
        raise RuntimeError(msg)

    s = summary(out)
    assert s["status"] == FAILED
    assert s["error"] == "RuntimeError: diverged"
    # what was measured before the failure is kept, and labelled as such
    assert len(s["eval"]["evaluations"]) == 1


def test_a_clean_exit_with_a_short_evaluation_is_not_complete(tmp_path: Path) -> None:
    """Raising nothing is not enough: the run must have done what it says."""
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="has 1 episodes, config says 2"):
        with Run(out, config()) as run:
            log_evaluation(run, 0, 0, [1.0])

    assert summary(out)["status"] == FAILED


def test_a_run_without_an_evaluation_is_not_complete(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="no evaluation"):
        with Run(out, config()) as run:
            run.log_train_episode(episode=0, env_step=3, **outcome())

    assert summary(out)["status"] == FAILED


def test_a_closed_run_records_nothing_more(tmp_path: Path) -> None:
    with Run(tmp_path / "run", config()) as run:
        log_evaluation(run, 0, 0, [1.0, 1.0])
    with pytest.raises(RuntimeError, match="closed"):
        run.log_train_episode(episode=0, env_step=1, **outcome())
    with pytest.raises(RuntimeError, match="already closed"):
        run.finish()


def test_the_wall_clock_budget_is_enforced(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(BudgetExceededError, match="max_wall_clock_s"):
        with Run(out, config(max_wall_clock_s=1e-9)) as run:
            run.check_budget()

    assert summary(out)["status"] == FAILED


def test_the_default_directory_names_algorithm_env_and_seed() -> None:
    assert default_out_dir(config(seed=3)) == Path(
        "runs", "rl", "test", "Test-v0", "seed-3"
    )
    assert default_out_dir(config(out_dir="here")) == Path("here")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
