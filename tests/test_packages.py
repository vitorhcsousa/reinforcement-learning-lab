"""The repository rename moved no import and no command."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

import pytest

RL_LAB = (
    "rl_lab",
    "rl_lab.bandits",
    "rl_lab.mdp",
    "rl_lab.dynamic_programming",
    "rl_lab.model_free",
    "rl_lab.value_based",
    "rl_lab.policy_gradient",
)

RLVR = (
    "verifiable_reward_lab",
    "verifiable_reward_lab.model",
    "verifiable_reward_lab.tokenizer",
    "verifiable_reward_lab.data",
    "verifiable_reward_lab.training",
    "verifiable_reward_lab.evaluation",
    "verifiable_reward_lab.rewards",
    "verifiable_reward_lab.rollout",
)

COMMANDS = {
    "rlvr-train": "verifiable_reward_lab.cli:main",
    "rlvr-eval": "verifiable_reward_lab.eval_cli:main",
    "rlvr-compare": "verifiable_reward_lab.reproduce:main",
}


@pytest.mark.parametrize("name", RL_LAB + RLVR)
def test_package_imports(name: str) -> None:
    importlib.import_module(name)


def test_rl_lab_placeholders_export_nothing() -> None:
    """Placeholders stay placeholders until their milestone lands with tests.

    Delete a subpackage from this check in the commit that implements it.
    """
    # importing a subpackage binds its name on the parent; that is not code
    subpackages = {name.rsplit(".", 1)[-1] for name in RL_LAB[1:]}
    for name in RL_LAB:
        module = importlib.import_module(name)
        public = {n for n in vars(module) if not n.startswith("_")}
        assert public <= subpackages, f"{name} exports {public - subpackages}"


@pytest.mark.parametrize(("command", "target"), COMMANDS.items())
def test_cli_entry_points_are_stable(command: str, target: str) -> None:
    (point,) = entry_points(group="console_scripts", name=command)
    assert point.value == target
    assert callable(point.load())
