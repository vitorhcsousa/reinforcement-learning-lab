"""A RunConfig that cannot describe a run is refused, and one that can survives a round trip."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from rl_lab.experiment.config import VERSION, RunConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def payload(**overrides: Any) -> dict[str, Any]:
    """A valid config as a dict, with any field replaced."""
    base: dict[str, Any] = {
        "version": VERSION,
        "env_id": "RandomWalk-v0",
        "algorithm_id": "scripted_walk",
        "seed": 7,
        "train_steps": 100,
        "eval_interval": 50,
        "eval_episodes": 4,
        "max_episode_steps": 10,
        "env_params": {"length": 4, "start": 2, "slip": 0.1},
        "algorithm_params": {"train_p_right": 0.5, "eval_p_right": 1.0},
        "max_wall_clock_s": 5.0,
        "device": "cpu",
        "out_dir": None,
    }
    base.update(overrides)
    return base


def test_dict_round_trip_is_the_same_config() -> None:
    config = RunConfig.from_dict(payload())
    assert RunConfig.from_dict(config.to_dict()) == config


def test_yaml_round_trip_is_the_same_config(tmp_path: Path) -> None:
    config = RunConfig.from_dict(payload(out_dir="runs/x"))
    path = tmp_path / "config.yaml"
    config.save(path)

    loaded = RunConfig.load(path)

    assert loaded == config
    assert loaded.sha256() == config.sha256()


def test_the_round_trip_keeps_types_not_just_values(tmp_path: Path) -> None:
    """0.1 must come back a float and True a bool, or 'same config' is luck."""
    config = RunConfig.from_dict(
        payload(algorithm_params={"a": 0.1, "b": True, "c": 3})
    )
    path = tmp_path / "config.yaml"
    config.save(path)

    params = RunConfig.load(path).algorithm_params

    assert params == {"a": 0.1, "b": True, "c": 3}
    assert [type(v) for v in params.values()] == [float, bool, int]


def test_an_integer_wall_clock_limit_is_the_same_limit() -> None:
    assert RunConfig.from_dict(payload(max_wall_clock_s=5)) == RunConfig.from_dict(
        payload(max_wall_clock_s=5.0)
    )


def test_the_committed_smoke_config_loads() -> None:
    config = RunConfig.load(REPO_ROOT / "configs" / "rl_smoke.yaml")
    assert config.env_id == "RandomWalk-v0"
    assert config.device == "cpu"


def test_identity_ignores_where_the_run_is_written() -> None:
    here = RunConfig.from_dict(payload(out_dir="runs/a"))
    there = dataclasses.replace(here, out_dir="elsewhere/b")
    assert here.sha256() == there.sha256()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 8),
        ("train_steps", 101),
        ("max_episode_steps", 11),
        ("env_params", {"length": 4, "start": 2, "slip": 0.2}),
        ("algorithm_params", {"train_p_right": 0.6, "eval_p_right": 1.0}),
    ],
)
def test_identity_changes_with_anything_that_moves_the_numbers(
    field: str, value: object
) -> None:
    base = RunConfig.from_dict(payload())
    assert RunConfig.from_dict(payload(**{field: value})).sha256() != base.sha256()


def test_the_config_does_not_alias_the_callers_params() -> None:
    params = {"length": 4, "start": 2, "slip": 0.1}
    config = RunConfig.from_dict(payload(env_params=params))
    params["slip"] = 0.9
    assert config.env_params["slip"] == 0.1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"train_steps": -1}, "train_steps must be >= 0"),
        ({"eval_episodes": 0}, "eval_episodes must be >= 1"),
        ({"eval_interval": 0}, "eval_interval must be >= 1"),
        ({"max_episode_steps": 0}, "max_episode_steps must be >= 1"),
        ({"train_steps": 1.5}, "train_steps must be an integer"),
        ({"train_steps": True}, "train_steps must be an integer"),
        ({"seed": -1}, r"seed must be in \[0, 4294967295\]"),
        ({"seed": 2**32}, r"seed must be in \[0, 4294967295\]"),
        ({"seed": 1.0}, "seed must be an integer"),
        ({"seed": True}, "seed must be an integer"),
        ({"seed": "7"}, "seed must be an integer"),
        ({"max_wall_clock_s": 0}, "max_wall_clock_s must be a finite number > 0"),
        ({"max_wall_clock_s": -5.0}, "max_wall_clock_s must be a finite number > 0"),
        (
            {"max_wall_clock_s": float("inf")},
            "max_wall_clock_s must be a finite number > 0",
        ),
        ({"device": "tpu"}, "device must be one of"),
        ({"env_id": "RandomWalk"}, "env_id must look like 'Name-v0'"),
        ({"algorithm_id": "Scripted Walk"}, "algorithm_id must be snake_case"),
        ({"env_params": {"nested": {"a": 1}}}, "env_params.nested must be a scalar"),
        (
            {"algorithm_params": {"lr": float("nan")}},
            "algorithm_params.lr must be finite",
        ),
        ({"out_dir": ""}, "out_dir must be a non-empty string or null"),
    ],
)
def test_invalid_values_are_refused_with_a_reason(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        RunConfig.from_dict(payload(**overrides))


@pytest.mark.parametrize(
    "field",
    [
        "env_id",
        "algorithm_id",
        "seed",
        "train_steps",
        "eval_interval",
        "eval_episodes",
        "max_episode_steps",
    ],
)
def test_a_missing_required_field_is_named(field: str) -> None:
    incomplete = payload()
    del incomplete[field]
    with pytest.raises(ValueError, match=f"missing required field.*{field}"):
        RunConfig.from_dict(incomplete)


def test_an_unknown_field_is_refused_not_ignored() -> None:
    """A typo would otherwise fall back to the default without a word."""
    with pytest.raises(ValueError, match="unknown field.*eval_episode$"):
        RunConfig.from_dict(payload(eval_episode=10))


def test_a_missing_or_foreign_version_is_refused() -> None:
    no_version = payload()
    del no_version["version"]
    with pytest.raises(ValueError, match="has no 'version'"):
        RunConfig.from_dict(no_version)
    with pytest.raises(ValueError, match="version 2 cannot be read"):
        RunConfig.from_dict(payload(version=2))


def test_a_yaml_file_that_is_not_a_mapping_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        RunConfig.load(path)
