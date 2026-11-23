"""The smoke experiment end to end: same seed same bytes, known answer, CPU only."""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from rl_lab.experiment.config import RunConfig
from rl_lab.experiment.run import VOLATILE
from rl_lab.experiment.smoke import RandomWalk, main, reach_probability, run_smoke

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE = REPO_ROOT / "configs" / "rl_smoke.yaml"


def smoke(**overrides: Any) -> RunConfig:
    return dataclasses.replace(RunConfig.load(SMOKE), **overrides)


def stable(run_dir: Path) -> dict[str, Any]:
    """summary.json without the fields that differ between identical runs."""
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for key in VOLATILE:
        del s[key]
    return s


def test_the_same_seed_reproduces_the_raw_outputs_byte_for_byte(
    tmp_path: Path,
) -> None:
    run_smoke(smoke(), tmp_path / "a")
    run_smoke(smoke(), tmp_path / "b")

    for name in ("config.yaml", "metrics.jsonl"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()
    assert stable(tmp_path / "a") == stable(tmp_path / "b")


def test_a_different_seed_changes_the_raw_outputs(tmp_path: Path) -> None:
    a = run_smoke(smoke(), tmp_path / "a")
    b = run_smoke(smoke(seed=1338), tmp_path / "b")

    assert a["config_sha256"] != b["config_sha256"]
    assert a["files"]["metrics"]["sha256"] != b["files"]["metrics"]["sha256"]
    # the training stream changed, so the training episodes did
    assert a["train"]["mean_return"] != b["train"]["mean_return"]


def test_evaluation_does_not_depend_on_how_long_training_ran(tmp_path: Path) -> None:
    """A scripted policy does not learn, so every evaluation of a run - and
    the first evaluation of runs with different budgets - is the same set of
    episodes. If training randomness leaked into evaluation, it would not be."""
    short = run_smoke(smoke(train_steps=500), tmp_path / "short")
    long = run_smoke(smoke(train_steps=2000), tmp_path / "long")

    assert short["eval"]["final"]["returns"] == long["eval"]["final"]["returns"]
    means = {e["mean_return"] for e in long["eval"]["evaluations"]}
    assert len(means) == 1


def test_training_does_not_depend_on_how_often_it_is_evaluated(
    tmp_path: Path,
) -> None:
    """The other half of isolation: evaluating must not advance the training
    environment or the training policy stream."""
    sparse = run_smoke(smoke(eval_interval=2000), tmp_path / "sparse")
    dense = run_smoke(smoke(eval_interval=100), tmp_path / "dense")

    def train_records(run_dir: Path) -> list[str]:
        lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        return [line for line in lines if '"kind": "train_episode"' in line]

    assert len(dense["eval"]["evaluations"]) > len(sparse["eval"]["evaluations"])
    assert train_records(tmp_path / "dense") == train_records(tmp_path / "sparse")


def test_the_training_environment_stream_continues_across_episodes(
    tmp_path: Path,
) -> None:
    """Seeded once, at the first reset. Reseeding every episode would replay
    the same noise; with a deterministic policy every episode would repeat."""
    s = run_smoke(
        smoke(algorithm_params={"train_p_right": 1.0, "eval_p_right": 0.9}),
        tmp_path / "run",
    )
    lines = (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8")
    episodes = [
        (r["return"], r["length"])
        for r in map(json.loads, lines.splitlines())
        if r["kind"] == "train_episode"
    ]

    assert s["train"]["episodes"] > 10
    assert len(set(episodes)) > 1


def test_training_and_evaluation_returns_are_reported_separately(
    tmp_path: Path,
) -> None:
    s = run_smoke(smoke(), tmp_path / "run")
    config = RunConfig.load(SMOKE)

    assert s["status"] == "complete"
    assert s["train"]["episodes"] > 0
    # the two policies differ, and so do their returns
    assert s["train"]["mean_return"] < s["eval"]["final"]["mean_return"]
    # evaluations at 0, every interval, and the end
    assert [e["env_step"] for e in s["eval"]["evaluations"]] == list(
        range(0, config.train_steps + 1, config.eval_interval)
    )
    assert all(e["episodes"] == config.eval_episodes for e in s["eval"]["evaluations"])


def test_the_committed_config_exercises_both_episode_endings(tmp_path: Path) -> None:
    """terminated and truncated are different events; the smoke run must
    produce both, or the code that separates them is untested."""
    s = run_smoke(smoke(), tmp_path / "run")
    assert s["train"]["terminated"] > 0
    assert s["train"]["truncated"] > 0


def test_an_off_interval_budget_still_evaluates_at_the_end(tmp_path: Path) -> None:
    s = run_smoke(smoke(train_steps=1250, eval_interval=500), tmp_path / "run")
    assert [e["env_step"] for e in s["eval"]["evaluations"]] == [0, 500, 1000, 1250]


def test_an_evaluation_only_run_is_a_run(tmp_path: Path) -> None:
    s = run_smoke(smoke(train_steps=0), tmp_path / "run")
    assert s["train"]["episodes"] == 0
    assert s["train"]["mean_return"] is None
    assert [e["env_step"] for e in s["eval"]["evaluations"]] == [0]


@pytest.mark.parametrize("p_right", [0.5, 0.8, 1.0])
def test_the_measured_return_matches_the_closed_form(
    tmp_path: Path, p_right: float
) -> None:
    """The experiment has a known answer, and the pipeline recovers it."""
    params = {"length": 6, "start": 3, "slip": 0.2}
    config = smoke(
        train_steps=0,
        eval_episodes=4000,
        max_episode_steps=10_000,  # long enough that truncation is negligible
        env_params=params,
        algorithm_params={"train_p_right": 0.5, "eval_p_right": p_right},
    )
    s = run_smoke(config, tmp_path / "run")

    expected = reach_probability(p_right, 0.2, length=6, start=3)
    measured = s["eval"]["final"]["mean_return"]
    stderr = math.sqrt(expected * (1 - expected) / config.eval_episodes)
    assert abs(measured - expected) < 4 * stderr
    assert s["eval"]["evaluations"][0]["truncated"] == 0


def test_the_closed_form_has_the_textbook_values() -> None:
    assert reach_probability(0.5, 0.0, length=6, start=3) == 0.5
    assert reach_probability(1.0, 0.0, length=6, start=3) == 1.0
    assert reach_probability(0.0, 0.0, length=6, start=3) == 0.0
    # q = 0.8, r = 1/4: (1 - 4^-3) / (1 - 4^-6)
    assert math.isclose(
        reach_probability(1.0, 0.2, length=6, start=3),
        (1 - 0.25**3) / (1 - 0.25**6),
    )


def test_the_walk_truncates_at_the_limit_and_terminated_wins() -> None:
    env = RandomWalk(length=4, start=2, slip=0.0, max_episode_steps=2)
    env.reset(seed=0)
    assert env.step(1)[2:4] == (False, False)
    # position 4 on the last allowed step: terminal, not cut off
    assert env.step(1)[1:4] == (1.0, True, False)

    env.reset()
    env.step(1)
    assert env.step(-1)[2:4] == (False, True)


def test_a_config_for_another_experiment_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runs scripted_walk on RandomWalk-v0"):
        run_smoke(smoke(env_id="Other-v0"), tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_the_cli_refuses_to_overwrite_a_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    assert main(["--config", str(SMOKE), "--out", str(out)]) == 0
    assert capsys.readouterr().out.endswith("OK\n")
    first = (out / "metrics.jsonl").read_bytes()

    assert main(["--config", str(SMOKE), "--out", str(out), "--seed", "1"]) == 2
    assert "already exists" in capsys.readouterr().err
    assert (out / "metrics.jsonl").read_bytes() == first


def test_the_cli_seed_override_is_what_gets_saved(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert main(["--config", str(SMOKE), "--out", str(out), "--seed", "9"]) == 0
    saved = RunConfig.load(out / "config.yaml")
    assert saved.seed == 9
    assert saved.out_dir == str(out)


def test_the_smoke_experiment_needs_no_model_stack(tmp_path: Path) -> None:
    """CPU only, no downloads: running it imports neither torch nor the
    transformer / SFT / GRPO pipeline nor anything that reaches the hub."""
    script = (
        "import sys\n"
        "from rl_lab.experiment.smoke import main\n"
        f"code = main(['--config', {str(SMOKE)!r}, '--out', {str(tmp_path / 'run')!r}])\n"
        "heavy = {'torch', 'transformers', 'datasets', 'huggingface_hub',"
        " 'verifiable_reward_lab'}\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))\n"
        "sys.exit(code)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={"HF_HUB_OFFLINE": "1", "PATH": ""},
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "[]"
