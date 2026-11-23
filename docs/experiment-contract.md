# Experiment contract

Every `rl_lab` run follows this contract, so that a result can be reproduced,
compared and audited by someone who was not there. The code that enforces it
is [`rl_lab/experiment/`](../src/rl_lab/experiment/); the tests that pin it are
in [`tests/rl_lab/experiment/`](../tests/rl_lab/experiment/).

The question it answers: *what would another engineer need to reproduce this
result without asking me anything?* A config that fully determines the run,
the code at a known commit, one command, and raw outputs to compare against.

The verifiable-rewards study predates this contract and keeps its own
(`rlvr-train`, `runs/tiny/`, `rlvr-compare`). The two are deliberately
independent.

## the config — `RunConfig`

A frozen dataclass, loaded from YAML. A config that cannot describe a run is
refused when it is built, with the field and the reason. Unknown keys are
refused rather than ignored, so a typo cannot fall back to a default.

| Field | Type | Required | Valid | Default |
| :-- | :-- | :-- | :-- | :-- |
| `version` | int | yes | `1` | — |
| `env_id` | str | yes | `Name-vN`; a change in dynamics is a new `N` | — |
| `algorithm_id` | str | yes | snake_case | — |
| `seed` | int | yes | `0 ≤ seed ≤ 2³² − 1`; bools and floats refused | — |
| `train_steps` | int | yes | `≥ 0` environment steps; `0` is an evaluation-only run | — |
| `eval_interval` | int | yes | `≥ 1` environment steps | — |
| `eval_episodes` | int | yes | `≥ 1` | — |
| `max_episode_steps` | int | yes | `≥ 1`; the time limit, part of the problem | — |
| `env_params` | map of scalars | no | flat; floats finite | `{}` |
| `algorithm_params` | map of scalars | no | flat; floats finite | `{}` |
| `max_wall_clock_s` | float | no | finite, `> 0` | `600.0` |
| `device` | str | no | `cpu`, `mps`, `cuda` | `cpu` |
| `out_dir` | str or null | no | non-empty | `runs/rl/<algorithm>/<env>/seed-<n>` |

**Identity.** Everything except `out_dir` is the run's identity, hashed as
`config_sha256` (sha256 of the canonical JSON). Moving a run does not change
it; changing anything that moves the numbers does.

## randomness

One seed is split with `numpy.random.SeedSequence(seed).spawn(3)` into:

| Stream | Consumed by |
| :-- | :-- |
| training environment | the first `reset(seed=…)` of the training env; later resets continue it |
| policy | action sampling during training |
| evaluation | `eval_episodes` episode seeds, **reused by every evaluation of the run** |

Each evaluation episode resets the evaluation env with its episode seed and
draws its actions from a child of that seed. Consequences, each pinned by a test:

- **evaluation never depends on training**: not on how long it ran, nor on how
  many random numbers the algorithm drew;
- **training never depends on evaluation**: evaluating more often leaves every
  training record unchanged;
- **comparisons are paired**: two algorithms run with the same seed are
  evaluated on the same episodes, and two evaluations within a run differ only
  because the policy did.

Python's `random` is seeded; torch only when a run asks for it. NumPy's global
state is deliberately left unseeded: code that reaches for it fails the
same-seed test instead of passing it by accident.

**Determinism is promised on one machine with one lockfile**, and tested as
byte-identical `metrics.jsonl`. It is not promised across machines: NumPy keeps
bit-generator streams stable across releases but not every distribution method
(NEP 19), and torch kernels differ across devices and builds. Agreement across
platforms is something to measure and report, not assume.

## the run directory

```text
<out_dir>/
├── config.yaml     # written before the first step: the run that happened, overrides included
├── metrics.jsonl   # one raw record per finished episode
└── summary.json    # derived from metrics.jsonl on disk, written last, atomically
```

- **Never reused.** The directory is created exclusively; if it exists, even
  empty, the run is refused before anything is written. A rerun cannot
  overwrite or mix into earlier evidence.
- **Failures are labelled.** A run that raises, exceeds `max_wall_clock_s`, or
  exits cleanly without the evaluations its config promises is closed with
  `status: failed` and the error. What it measured before failing is kept.
- **Not committed.** `runs/` is git-ignored. Evidence that a claim rests on is
  copied to [`experiments/`](../experiments/README.md).

### `metrics.jsonl`

One JSON object per line, written with `repr` floats (exact round trip, no
rounding) and `allow_nan=False` (a non-finite value fails loudly). Only finished
episodes are recorded: the episode in progress when the training budget runs
out has no return, and is dropped rather than recorded as a short one.

| Field | `train_episode` | `eval_episode` |
| :-- | :-- | :-- |
| `kind` | `"train_episode"` | `"eval_episode"` |
| `evaluation` | — | 0-based index of the evaluation |
| `env_step` | training steps when the episode finished | training steps when the evaluation ran |
| `episode` | 0-based among training episodes | 0-based within the evaluation |
| `seed` | — | the episode's reset seed, to replay it alone |
| `return` | undiscounted sum of rewards | same |
| `length` | steps, `≥ 1` | same |
| `terminated` | reached a terminal state | same |
| `truncated` | cut by the time limit without reaching one | same |

No wall-clock time is recorded per episode, so that two runs of the same
config produce the same bytes.

### `summary.json`

| Key | Content |
| :-- | :-- |
| `status`, `error` | `complete` or `failed`, and why |
| `env_id`, `algorithm_id`, `seed` | what was run |
| `config_sha256`, `config` | identity hash and the full config |
| `files` | path and sha256 of `config.yaml` and `metrics.jsonl`, and the record count |
| `train` | episodes, mean return, terminated and truncated counts |
| `eval` | episodes per evaluation; per evaluation: env step, episodes, mean return, terminated and truncated counts; for the final one, every raw return |
| `versions`, `platform` | python, numpy, pyyaml, torch, the package |
| `git` | commit, and whether the working tree had local changes |
| `wall_clock_s`, `started_at`, `finished_at` | the only keys allowed to differ between identical runs |

Means use `math.fsum`. No standard error or interval is computed inside a run:
uncertainty is a property of a set of runs, see below.

## episode endings

`terminated` and `truncated` are different events and are recorded separately.
A terminated episode reached a terminal state: its return is a true return. A
truncated one was cut by `max_episode_steps`: its return is a lower bound for
a task with positive rewards, and an algorithm that bootstraps must bootstrap
from its last state rather than treat it as terminal. If a terminal state is
reached on the last allowed step, the episode is terminated, not truncated.

`max_episode_steps` belongs to the problem, not the algorithm: two runs with
different limits are not comparable.

## protocol for a comparison

What every `rl_lab` comparison states before it reports a number:

- **Baseline.** Name it, and give it the same tuning budget as the method. A
  tuned method against an untuned baseline measures the tuning. The simplest
  defensible baseline (random, or the scripted policy) is reported beside the
  strongest one.
- **No evaluation leakage.** Hyperparameters are chosen on evaluation seeds
  that are not the ones reported, or with the reported ones fixed before any
  result is seen. The final number is never the best evaluation of a run
  picked after the fact; it is the evaluation the protocol named in advance
  (by default, the last).
- **Seeds.** The seed count is fixed before running. Comparisons use the same
  seeds for every arm, so differences are paired. Three seeds give a spread,
  not a confidence interval; a claim that needs an interval needs more runs.
- **Uncertainty.** Reported across runs, never within one: a bootstrap
  confidence interval over per-seed results, next to the per-seed values
  themselves. For many tasks or seeds, interquartile mean with stratified
  bootstrap intervals (Agarwal et al., 2021). A difference smaller than the
  spread across seeds is reported as such.
- **Compute.** Wall clock, hardware and environment steps are reported with
  the result. A comparison at unequal environment steps says so.
- **Deviations.** A config change after a result has been seen is recorded
  with its date and reason.

## reproducing the smoke experiment from a clean clone

```bash
git clone https://github.com/vitorhcsousa/reinforcement-learning-lab.git
cd reinforcement-learning-lab
uv sync --frozen --group dev
make rl-smoke                                                        # seed 1337
uv run python -m rl_lab.experiment.smoke --out runs/rl/repeat        # same seed again
uv run python -m rl_lab.experiment.smoke --out runs/rl/other --seed 1338

cmp runs/rl/scripted_walk/RandomWalk-v0/seed-1337/metrics.jsonl runs/rl/repeat/metrics.jsonl  # identical
cmp runs/rl/scripted_walk/RandomWalk-v0/seed-1337/metrics.jsonl runs/rl/other/metrics.jsonl   # differs
```

No data, no model download, no accelerator; each run takes well under a
second. Running `make rl-smoke` a second time is refused, by design. The
experiment is a scripted random walk whose expected evaluation return has a
closed form (0.95843 for the committed config with no time limit), checked by
the tests.

## references

- Patterson, Neumann, White and White. [Empirical Design in Reinforcement Learning](https://www.jmlr.org/papers/v25/23-0183.html). JMLR 25, 2024.
- Agarwal, Schwarzer, Castro, Courville and Bellemare. [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264). NeurIPS 2021.
- Henderson et al. [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560). AAAI 2018.
- [Gymnasium: terminated vs truncated](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
- [NEP 19 — Random number generator policy](https://numpy.org/neps/nep-0019-rng-policy.html).
