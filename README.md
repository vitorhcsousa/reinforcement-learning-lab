# Reinforcement Learning Lab

[![CI](https://github.com/vitorhcsousa/reinforcement-learning-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/vitorhcsousa/reinforcement-learning-lab/actions)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Reinforcement learning built from first principles and measured honestly:
> foundational algorithms implemented from scratch, and finite empirical
> studies with frozen protocols, seed variability and failure analysis.

## About

This repository is a research and engineering lab for reinforcement learning.
It holds two kinds of work, kept apart on purpose:

- **Implementations** — algorithms written from raw tensors and arrays, one
  file per concept, readable top to bottom, with tests that pin the properties
  that fail silently. General RL lives in `rl_lab`; the transformer, data and
  training stack built for post-training lives in `verifiable_reward_lab`.
- **Studies** — bounded empirical questions with a protocol frozen before any
  result is seen. The first, [Learning with Verifiable Rewards](studies/verifiable-rewards/README.md),
  asks whether GRPO earns its cost against a properly built SFT baseline on
  GSM8K, and where it fails.

Until 2026-11-22 the repository was `verifiable-reward-lab` and held only the
second study. It was renamed and widened, not replaced: the history, the
`verifiable_reward_lab` package and the `rlvr-*` commands are unchanged.

This is not a tutorial and not an RL framework.

## Research Questions and Principles

Each study owns its questions. The active ones:

| Study | Question |
| :-- | :-- |
| Verifiable rewards · RQ1 | Does GRPO materially improve held-out GSM8K accuracy over the chosen SFT baseline at this scale? |
| Verifiable rewards · RQ2 | How sensitive is the result to group size? |
| Verifiable rewards · RQ3 | What effect does KL regularization have? |
| Verifiable rewards · RQ5 | Where does reward improve without a corresponding improvement in held-out task accuracy? |

RQ4 (format reward) was retired before any GRPO run: the base model already
scores 200/200 on format, so there is nothing for that ablation to move.
[Details](studies/verifiable-rewards/README.md#research-questions).

Principles, for every study and every module:

- **Implemented is not verified.** Code with passing unit tests is implemented.
  A claim about behaviour is verified only when a committed config reproduces it.
- **Measure, don't assume.** A point estimate without seed variability is not a
  result. Reward improving is not accuracy improving until it has been checked.
- **Reward over optimizer.** A better optimizer only exploits a flawed reward
  faster; the scrutiny goes to the verifier.
- **Freeze before you look.** Hyperparameters and evaluation sets are fixed
  before results exist. A change after that is a dated deviation.
- **Finite by design.** One variable per ablation; no algorithm added for breadth.

## Learning and Implementation Roadmap

| Track | Stages | Status |
| :-- | :-- | :-- |
| Foundations | transformer, KV-cache, sampling, tokenizer, data, training loop | verified |
| General RL (`rl_lab`) | experiment contract → finite MDP → dynamic programming → bandits → model-free → value-based → policy gradient | **experiment contract done; next: finite MDP** |
| Post-training (study) | verifier → evaluation → SFT → group rollout → GRPO → ablations | verifier and evaluation done; SFT partial; rollout and GRPO not started |

Full roadmap, with what "done" means for each stage: [docs/roadmap.md](docs/roadmap.md).

## Repository Structure

```text
src/
├── rl_lab/                    # general RL
│   ├── experiment/            #   run contract: config, seeds, run directory, smoke experiment
│   ├── mdp/                   #   finite MDPs                       (next)
│   ├── dynamic_programming/   #   policy / value iteration
│   ├── bandits/               #   multi-armed and contextual
│   ├── model_free/            #   Monte Carlo, TD, SARSA, Q-learning
│   ├── value_based/           #   function approximation, DQN
│   └── policy_gradient/       #   REINFORCE, actor-critic
└── verifiable_reward_lab/     # the verifiable-rewards study
    ├── model/                 #   decoder-only transformer from raw tensors
    ├── tokenizer/             #   character-level tokenizer
    ├── data/                  #   pinned corpora, GSM8K split, batching
    ├── training/              #   tiny-run trainer, SFT, frozen GRPO config
    ├── evaluation/            #   GSM8K evaluation harness
    ├── rewards/               #   verifier
    ├── rollout/               #   group sampling and advantages   (not implemented)
    ├── cli.py                 #   rlvr-train
    ├── eval_cli.py            #   rlvr-eval
    └── reproduce.py           #   rlvr-compare
configs/                       # every committed run starts from one of these
tests/                         # mirrors src/
benchmarks/                    # KV-cache throughput
notebooks/                     # interactive explorations (marimo)
studies/                       # one README per study: question, protocol, status
experiments/                   # committed experiment records (empty so far)
docs/roadmap.md
```

`runs/`, `checkpoints/` and `data/` are produced locally and git-ignored.

## Current Studies

| Study | Status | Page |
| :-- | :-- | :-- |
| **Learning with Verifiable Rewards** — GRPO vs SFT on GSM8K with Qwen 2.5-0.5B | in progress: verifier, evaluation and one SFT seed done; GRPO not implemented | [studies/verifiable-rewards](studies/verifiable-rewards/README.md) |
| **Contextual Bandits** | no verified code: a published article series whose code blocks have never run under test here | [studies/contextual-bandits](studies/contextual-bandits/README.md) |

## Implementation and Evidence Status

As of 2026-11-22. **Implemented** means code with passing tests. **Run** means
a saved local result exists. **Verified** means reproduced from a committed
config.

| Area | Status | Evidence |
| :-- | :-- | :-- |
| Transformer: attention, RoPE / sinusoidal / learned / ALiBi, RMSNorm / LayerNorm, SwiGLU / GELU, pre/post-norm blocks | **Implemented**, tests pass | 126 tests in `tests/model/`, incl. causality and finite-difference gradient checks |
| KV-cache | **Implemented**, tests pass | incremental decoding ≡ full forward at `atol=1e-5`, across positional encodings |
| Sampling: greedy / temperature / top-k / top-p | **Implemented**, tests pass | `tests/model/test_sampling.py` |
| Data pipeline: tokenizer, pinned corpora, GSM8K split, batching | **Implemented**, tests pass | 95 tests in `tests/data/`, `tests/tokenizer/` |
| Verifier | **Implemented**, tests pass | 37 tests in `tests/rewards/` |
| Tiny reference run | **Verified on one machine** | `configs/tiny.yaml` reproduced locally on 2026-11-22 at **1.7677081286907197 nats**, bit-identical to the saved run |
| GSM8K evaluation | **Implemented, run**; comparison not yet controlled | base model 4-shot 0.410 and SFT 0-shot 0.445 on dev[:200] — different shot counts by design, one seed, and not the RQ1 comparison |
| SFT baseline | **Implemented, run**; experimental validation incomplete | **one seed (1337) of the three RQ1 requires** |
| RLVR group rollout | **Not implemented** | `rollout/` is a placeholder with no tests |
| GRPO training | **Not implemented** | only the frozen hyperparameters, `configs/grpo_base.yaml` |
| Contextual bandits | **Blog code snippets only; unverified** | [studies/contextual-bandits](studies/contextual-bandits/README.md) |
| Experiment contract (`rl_lab.experiment`) | **Implemented**, tests pass; smoke run **verified from a clean checkout** | [docs/experiment-contract.md](docs/experiment-contract.md); same seed → byte-identical `metrics.jsonl`, measured return matches the closed form |
| General RL algorithms (`rl_lab`) | **Not implemented** | empty placeholder packages |
| CI | **green on `master`** | ruff, ty and pytest; 594 tests pass |

## Getting Started

```bash
git clone https://github.com/vitorhcsousa/reinforcement-learning-lab.git
cd reinforcement-learning-lab
uv sync --group dev
make data    # download + checksum-verify the pinned corpora
make check   # ruff + ty + pytest
```

No corpus is versioned. `make data` fetches each from a pinned URL and verifies
a sha256 recorded in `src/verifiable_reward_lab/data/fetch.py`, failing loudly
if the bytes differ. It is idempotent. Two corpora are pinned: tinyshakespeare
for the reference run, and GSM8K at an upstream commit rather than a branch, so
the URL cannot move under the checksum.

```bash
uv run python -m verifiable_reward_lab.data.gsm8k   # GSM8K split counts + dev fingerprint
```

Generate from the model layer:

```python
import torch

from verifiable_reward_lab.model.transformer import DecoderTransformer, TransformerConfig

config = TransformerConfig(
    vocab_size=256, d_model=128, n_layers=4, n_heads=4, max_seq_len=256
)
model = DecoderTransformer(config)

prompt = torch.randint(0, 256, (1, 8))  # (B, T) token ids

greedy = model.generate(prompt, max_new_tokens=32)  # argmax, deterministic
sampled = model.generate(
    prompt,
    max_new_tokens=32,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    generator=torch.Generator().manual_seed(0),  # reproducible draws
)
```

Development commands:

```text
make data       # download + verify the pinned corpora
make train      # the tiny reference run
make reference  # the reference run twice, then check the two agree
make eval       # base model on GSM8K dev, from the committed config
make rl-smoke   # the general-RL smoke experiment (CPU, <1 s)
make check      # lint + type check + tests (what CI runs)
make test-cov   # tests with coverage
make format     # auto-format
```

## Reproducing Experiments

### the tiny reference run

```bash
uv run rlvr-train --config configs/tiny.yaml   # writes runs/tiny/
make reference                                 # runs it twice and compares the two summaries
```

The run fetches and checksums its own corpus; there is no manual data step. It
writes `config.yaml`, `tokenizer.json`, `metrics.jsonl`, `summary.json` and
`ckpt.pt`. `make reference` writes to `runs/ref-a` and `runs/ref-b` and
overwrites them.

| | |
| :-- | :-- |
| Hardware | CPU — no accelerator |
| Runtime | ≤15 minutes on CPU |
| Tolerance | the same seed reproduces the final validation loss within **±0.05 nats** |

The tolerance is a property of this run, not a universal constant, and it is
never widened to rescue a reproduction that failed.

| Machine | torch | wall clock | final val loss | Evidence |
| :-- | :-- | --: | --: | :-- |
| Apple Silicon, macOS, CPU | 2.11.0 | 35 s | 1.7677081286907197 | reproduced 2026-11-22 against the saved 2026-08-27 run; `rlvr-compare` Δ = 0.0000 |
| x86-64, Linux, CPU | 2.13.0 | 80 s | 1.7677 | reported 2026-08; no artifact kept |

Not yet verified: a run from a fresh clone with an empty `data/`.

### general RL: the smoke experiment

```bash
make rl-smoke   # writes runs/rl/scripted_walk/RandomWalk-v0/seed-1337/
```

Every `rl_lab` run follows the [experiment contract](docs/experiment-contract.md):
a validated config, split random streams, and a run directory that is never
reused, holding `config.yaml`, raw `metrics.jsonl` and a `summary.json` that
names the config hash, the metrics hash, the versions and the commit. The
smoke experiment is a scripted random walk with a closed-form expected return;
the contract page has the clean-clone reproduction procedure.

### the verifiable-rewards study

The SFT and evaluation commands, and what each run produced, are in
[studies/verifiable-rewards](studies/verifiable-rewards/README.md#running-it).
They download Qwen 2.5-0.5B and the committed configs target Apple Silicon (`mps`).

### KV-cache throughput

```bash
uv run python benchmarks/bench_kv_cache.py
```

Greedy decoding, batch 1, 16-token prompt; `d_model=256`, 4 layers, 4 heads,
vocab 1000, RoPE, float32, CPU, seed 0. Median of 5 runs, 1 warmup excluded.
Not re-run since it was first recorded.

| n_new | use_cache | median tok/s |    min-max    |
| :---: | :-------: | :----------: | :-----------: |
|  128  |   False   |    317.9     | 312.2 - 326.8 |
|  128  |   True    |    814.3     | 785.9 - 837.3 |
|  256  |   False   |    185.2     | 175.3 - 185.7 |
|  256  |   True    |    699.5     | 683.7 - 707.2 |

Without the cache each step reprocesses the whole sequence, O(T²) per
generation; with it, each step attends one token to the cached K/V, O(T). The
speedup widens from 2.6× at 128 tokens to 3.8× at 256.

## Research Notes and Publications

Transformer internals, on the code in `verifiable_reward_lab/model/`:

- **Part 1** · [Attention Is All You Need to Implement](https://www.vitorsousa.com/foundations/attention-from-scratch/) — scaled dot-product and multi-head attention
- **Part 2** · [Positional Encoding: Teaching Transformers to Count](https://www.vitorsousa.com/foundations/positional-encoding/) — sinusoidal, learned, RoPE, ALiBi
- **Part 3** · [Building a Transformer: The Complete Forward Pass](https://www.vitorsousa.com/foundations/building-a-transformer/) — norms, residuals, FFN, and the assembled decoder
- _Part 4: Training a Transformer — in progress_
- [A Transformer from Raw Tensors: What 112 Tests Taught Me](https://www.vitorsousa.com/blog/transformer-from-raw-tensors/) — the three properties that fail silently, and the tests that pin them. The count in the title is from publication; the model layer now has 126.

Contextual bandits, a five-part series from November 2025. Its code blocks are
not backed by tested code in this repository; see [the study page](studies/contextual-bandits/README.md).

- [When to Use Contextual Bandits: The Decision Framework](https://www.vitorsousa.com/blog/contextual-bandits/)
- [Contextual Bandit Theory: Regret Bounds and Exploration](https://www.vitorsousa.com/blog/contextual-bandits-math/)
- [Implementing Contextual Bandits: Complete Algorithm Guide](https://www.vitorsousa.com/blog/contextual-bandits-algorithms/)
- [Neural Contextual Bandits for High-Dimensional Data](https://www.vitorsousa.com/blog/contextual-bandits-neural/)
- [Deploying Contextual Bandits: Production Guide and Offline Evaluation](https://www.vitorsousa.com/blog/contextual-bandits-prod/)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (Holtzman et al., 2019)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (Sutton and Barto, 2nd ed., 2018)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (Shao et al., 2024) — introduces GRPO
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021) — GSM8K

## License

MIT
