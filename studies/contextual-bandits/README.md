# Study · Contextual Bandits

**Status: no verified code. Nothing in this repository implements a bandit yet.**

## what exists

A five-part article series, published November 2025:

| Part | Article | Python blocks |
| :-- | :-- | --: |
| 1 | [When to Use Contextual Bandits: The Decision Framework](https://www.vitorsousa.com/blog/contextual-bandits/) | 0 |
| 2 | [Contextual Bandit Theory: Regret Bounds and Exploration](https://www.vitorsousa.com/blog/contextual-bandits-math/) | 0 |
| 3 | [Implementing Contextual Bandits: Complete Algorithm Guide](https://www.vitorsousa.com/blog/contextual-bandits-algorithms/) | 9 |
| 4 | [Neural Contextual Bandits for High-Dimensional Data](https://www.vitorsousa.com/blog/contextual-bandits-neural/) | 6 |
| 5 | [Deploying Contextual Bandits: Production Guide and Offline Evaluation](https://www.vitorsousa.com/blog/contextual-bandits-prod/) | 13 |

## what does not

- **The standalone repository is gone.** The code the series was written from
  lived in a separate repository, which has since been deleted. It was not
  migrated here, and there is no saved run, result or test from it.
- **The 28 code blocks in the articles are not evidence.** They have never been
  run as code under test in this repository. They are illustrations of the
  method, and are treated as unverified until each one is re-implemented here
  with tests.

They are deliberately not copied into [`rl_lab/bandits/`](../../src/rl_lab/bandits/).
Pasting untested snippets into a package would make them look like
implemented code; the lab's rule is that a module exists when its tests do.

## the path to verified

Bandits are a stage of the [roadmap](../../docs/roadmap.md); the first
milestone is the finite MDP, not this. Rebuilding them here means: one file per algorithm in
`rl_lab/bandits/`, tests that pin the properties the articles claim (for
example, regret growth against a known optimal arm), and a reproducible
experiment whose `summary.json` lands in [`experiments/`](../../experiments/README.md).
Until then, the articles stay as writing, and this page stays as is.
