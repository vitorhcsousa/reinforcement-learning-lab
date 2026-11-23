"""The frozen description of one RL run: what another engineer needs to redo it.

Same contract as `verifiable_reward_lab.training.config.TrainConfig`: a run
that cannot happen is rejected on construction, and a config survives a YAML
round trip as the same object.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

from rl_lab.experiment.seeding import check_seed

if TYPE_CHECKING:
    from pathlib import Path

# bumped when a field changes meaning, not when one is added with a default
VERSION = 1

DEVICES = ("cpu", "mps", "cuda")

# gymnasium's convention: the version is part of the id, so an environment
# whose dynamics change gets a new id instead of silently new numbers
_ENV_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*-v\d+$")
_ALGORITHM_ID = re.compile(r"^[a-z][a-z0-9_]*$")

_REQUIRED = (
    "env_id",
    "algorithm_id",
    "seed",
    "train_steps",
    "eval_interval",
    "eval_episodes",
    "max_episode_steps",
)

type Scalar = bool | int | float | str


@dataclass(frozen=True)
class RunConfig:
    """Everything that decides a run's numbers, and where they are written."""

    # what is being run
    env_id: str
    """Versioned, `Name-vN`. Changing the dynamics means a new N."""

    algorithm_id: str
    """snake_case. Names the code that acts; its knobs go in algorithm_params."""

    seed: int
    """The only source of randomness. 0 <= seed < 2**32."""

    # the budget
    train_steps: int
    """Environment steps spent training. 0 is an evaluation-only run."""

    eval_interval: int
    """Environment steps between evaluations. Evaluation also runs at step 0
    and at the end, so the first and last numbers always exist."""

    eval_episodes: int
    """Episodes per evaluation. Every evaluation reuses the same episode seeds,
    so two evaluations differ only because the policy did."""

    max_episode_steps: int
    """The time limit. An episode cut here is truncated, not terminated. Part
    of the problem: two runs with different limits are different problems."""

    # the knobs
    env_params: dict[str, Scalar] = field(default_factory=dict)
    algorithm_params: dict[str, Scalar] = field(default_factory=dict)

    # resource limits
    max_wall_clock_s: float = 600.0
    """A run that exceeds this is stopped and marked failed, not truncated
    into a result."""

    device: str = "cpu"

    # where
    out_dir: str | None = None
    """Where the run is written. None derives runs/rl/<algorithm>/<env>/seed-<n>.
    Not part of the run's identity: moving a run does not change it."""

    def __post_init__(self) -> None:
        if not isinstance(self.env_id, str) or not _ENV_ID.match(self.env_id):
            msg = f"env_id must look like 'Name-v0', got {self.env_id!r}"
            raise ValueError(msg)
        if not isinstance(self.algorithm_id, str) or not _ALGORITHM_ID.match(
            self.algorithm_id
        ):
            msg = f"algorithm_id must be snake_case, got {self.algorithm_id!r}"
            raise ValueError(msg)

        check_seed(self.seed)

        _check_int("train_steps", self.train_steps, minimum=0)
        _check_int("eval_interval", self.eval_interval, minimum=1)
        _check_int("eval_episodes", self.eval_episodes, minimum=1)
        _check_int("max_episode_steps", self.max_episode_steps, minimum=1)

        _check_params("env_params", self.env_params)
        _check_params("algorithm_params", self.algorithm_params)

        limit = self.max_wall_clock_s
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int | float)
            or not math.isfinite(limit)
            or limit <= 0
        ):
            msg = f"max_wall_clock_s must be a finite number > 0, got {limit!r}"
            raise ValueError(msg)
        if self.device not in DEVICES:
            msg = f"device must be one of {DEVICES}, got {self.device!r}"
            raise ValueError(msg)
        if self.out_dir is not None and (
            not isinstance(self.out_dir, str) or not self.out_dir
        ):
            msg = f"out_dir must be a non-empty string or null, got {self.out_dir!r}"
            raise ValueError(msg)

        # an int limit from yaml is the same limit; store one type. the
        # params are copied so the caller's dict cannot move a frozen config
        object.__setattr__(self, "max_wall_clock_s", float(limit))
        object.__setattr__(self, "env_params", dict(self.env_params))
        object.__setattr__(self, "algorithm_params", dict(self.algorithm_params))

    def to_dict(self) -> dict[str, Any]:
        """Plain, YAML-safe dict, version first."""
        payload: dict[str, Any] = {"version": VERSION}
        payload.update(dataclasses.asdict(self))
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunConfig:
        """Inverse of to_dict, refusing what it does not understand.

        Raises:
            ValueError: Missing version or required field, a version this code
                cannot read, a key it does not know, or an invalid value.
        """
        payload = dict(payload)
        if "version" not in payload:
            msg = f"config has no 'version'; this code reads version {VERSION}"
            raise ValueError(msg)
        found = payload.pop("version")
        if found != VERSION:
            msg = f"config version {found!r} cannot be read by version {VERSION}"
            raise ValueError(msg)

        missing = [name for name in _REQUIRED if name not in payload]
        if missing:
            msg = f"config is missing required field(s): {', '.join(missing)}"
            raise ValueError(msg)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            # a typo'd key would otherwise fall back to a default, silently
            msg = f"config has unknown field(s): {', '.join(unknown)}"
            raise ValueError(msg)

        for name in ("env_params", "algorithm_params"):
            if payload.get(name) is None:
                payload.pop(name, None)
        return cls(**payload)

    def save(self, path: Path) -> None:
        """Write as YAML."""
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), "utf-8")

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        """Read a config written by `save`, or by hand.

        Raises:
            ValueError: The file is not a mapping, or from_dict refuses it.
        """
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"{path}: expected a YAML mapping, got {type(payload).__name__}"
            raise ValueError(msg)
        return cls.from_dict(payload)

    def identity(self) -> dict[str, Any]:
        """What makes two runs the same run: everything but where it is written."""
        payload = self.to_dict()
        del payload["out_dir"]
        return payload

    def sha256(self) -> str:
        """A fingerprint of `identity`, stable across key order and machines."""
        canonical = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_int(name: str, value: object, *, minimum: int) -> None:
    # bool is an int to python, and `train_steps: true` is a typo, not a 1
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer, got {value!r}"
        raise ValueError(msg)
    if value < minimum:
        msg = f"{name} must be >= {minimum}, got {value}"
        raise ValueError(msg)


def _check_params(name: str, params: object) -> None:
    if not isinstance(params, dict):
        msg = f"{name} must be a mapping, got {type(params).__name__}"
        raise ValueError(msg)
    for key, value in params.items():
        if not isinstance(key, str):
            msg = f"{name} keys must be strings, got {key!r}"
            raise ValueError(msg)
        if not isinstance(value, bool | int | float | str):
            # a nested structure would round-trip, but would not hash or
            # compare the way a reader expects; flat is the contract
            msg = f"{name}.{key} must be a scalar, got {type(value).__name__}"
            raise ValueError(msg)
        if isinstance(value, float) and not math.isfinite(value):
            msg = f"{name}.{key} must be finite, got {value}"
            raise ValueError(msg)
