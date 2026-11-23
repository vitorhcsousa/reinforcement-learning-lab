# Roadmap

From a finite MDP to RL with verifiable rewards, each stage built from scratch,
tested, and run before the next one starts. A stage is **done** when its code
has tests in `tests/`, `make check` is green on `master`, and any claim it makes
has a reproducible run behind it.

Statuses: **Verified** (tested and run) · **Implemented** (code and tests, no
experiment yet) · **Partial** · **Not started**.

## foundations — `verifiable_reward_lab`

Built for the verifiable-rewards study, and the base everything neural reuses.

| Stage | Status | Where |
| :-- | :-- | :-- |
| Decoder-only transformer: attention, positional encodings, norms, FFN, blocks | Verified | `model/` — 126 tests |
| KV-cache (≡ full forward at `atol=1e-5`) and sampling | Verified | `model/transformer.py`, `model/sampling.py` |
| Tokenizer, pinned data, training loop | Verified | `tokenizer/`, `data/`, `training/` |
| Tiny reference run | Verified, one machine | `configs/tiny.yaml` — 1.7677081286907197 nats, reproduced 2026-11-22 |

## general RL — `rl_lab`

| # | Stage | Module | Status |
| --: | :-- | :-- | :-- |
| 0 | Experiment contract — validated config, split seeds, never-reused run directory, smoke experiment with a closed-form answer | `rl_lab/experiment/` | Verified — [contract](experiment-contract.md) |
| 1 | **Finite MDP model** — states, actions, transition and reward tensors, a few classic environments | `rl_lab/mdp/` | **Next** |
| 2 | Dynamic programming — policy evaluation, policy iteration, value iteration | `rl_lab/dynamic_programming/` | Not started |
| 3 | Bandits — multi-armed, then contextual (ε-greedy, UCB, LinUCB, Thompson sampling) | `rl_lab/bandits/` | Not started — see [the study](../studies/contextual-bandits/README.md) |
| 4 | Model-free prediction and control — Monte Carlo, TD(0), SARSA, Q-learning | `rl_lab/model_free/` | Not started |
| 5 | Value-based with function approximation — linear, then DQN | `rl_lab/value_based/` | Not started |
| 6 | Policy gradient — REINFORCE, baselines, actor-critic, clipped surrogate | `rl_lab/policy_gradient/` | Not started |

Each stage is one file per concept, readable top to bottom, not a framework:
no environment registry, no trainer abstraction shared across algorithms.

## post-training — the verifiable-rewards study

| Stage | Status |
| :-- | :-- |
| Verifier | Implemented |
| GSM8K evaluation harness | Implemented, run |
| SFT baseline | Partial — 1 of 3 seeds |
| Group rollout and advantages | Not started |
| GRPO objective and loop | Not started |
| RQ1–RQ3, RQ5 experiments | Not started |

Protocol and research questions: [studies/verifiable-rewards](../studies/verifiable-rewards/README.md).

Stage 6 and the GRPO stage meet at the clipped surrogate objective. They stay
separate: the study implements the one GRPO it needs, and does not wait for, or
become, the general policy-gradient stage.

## not on the roadmap

DAPO, GDPO or further algorithms added for breadth; a generic RL framework or
RLHF library; large-scale pretraining; distributed training infrastructure;
agent frameworks unrelated to a study's question.
