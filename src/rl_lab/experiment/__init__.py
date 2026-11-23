"""The experiment contract every rl_lab run follows: config, seeds, run directory.

See docs/experiment-contract.md.
"""

from __future__ import annotations

from rl_lab.experiment.config import RunConfig
from rl_lab.experiment.run import (
    BudgetExceededError,
    Run,
    RunExistsError,
    default_out_dir,
)
from rl_lab.experiment.seeding import (
    Streams,
    check_seed,
    episode_policy_rng,
    make_streams,
    seed_libraries,
)

# `smoke` is deliberately not re-exported: it runs as
# `python -m rl_lab.experiment.smoke`, and importing it here would load it
# once as a package attribute and again as __main__.
__all__ = [
    "BudgetExceededError",
    "Run",
    "RunConfig",
    "RunExistsError",
    "Streams",
    "check_seed",
    "default_out_dir",
    "episode_policy_rng",
    "make_streams",
    "seed_libraries",
]
