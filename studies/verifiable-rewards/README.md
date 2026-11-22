# Study · Learning with Verifiable Rewards

Does GRPO earn its cost against a properly built SFT baseline at small scale,
and where does it fail?

This is the lab's first study, and it predates the lab: the repository was
called `verifiable-reward-lab` until 2026-11-22. Its code is the
`verifiable_reward_lab` package, its commands are `rlvr-train`, `rlvr-eval`
and `rlvr-compare`, and none of those were renamed with the repository.

It is a study, not an RL library. It implements one algorithm and varies it
along a fixed set of axes. It is finite by design.

## research questions

| | |
| :-- | :-- |
| **RQ1** | Does GRPO materially improve held-out GSM8K accuracy over the chosen SFT baseline at this scale? |
| **RQ2** | How sensitive is the result to group size? |
| **RQ3** | What effect does KL regularization have? |
| **RQ5** | Where does reward improve without a corresponding improvement in held-out task accuracy? |

**RQ4 (format reward) is retired.** It asked how format reward changes
optimization behaviour. The frozen base-model evaluation came back at 200/200
on format with zero no-answers, so there is no headroom for a format reward to
move and an ablation on it would measure zero. The decision was made in W36,
before any GRPO run, and is recorded in [`training/grpo_config.py`](../../src/verifiable_reward_lab/training/grpo_config.py).
The number is kept, not reused, so RQ5 stays RQ5.

## protocol

Frozen in W36, before a single GRPO number existed: a hyperparameter chosen
after seeing a result is a hyperparameter fitted to the result. Changing a
frozen config after a result has been seen is a deviation, and is recorded
with a date and a reason.

| | |
| :-- | :-- |
| Base model | Qwen 2.5-0.5B, pinned revision `060db649…` |
| Task | GSM8K. Train/dev cut is content-addressed and re-derivable from its seed (1337). Held-out is upstream's own test split, never re-cut from train |
| Pipeline | SFT baseline → verifier → group rollout → GRPO → held-out evaluation → ablations → failure analysis |
| Comparison | per question, on the same frozen dev prefix (n=200), so every delta is a paired quantity. At n=200 an unpaired interval is about ±7 points — wide enough to hide any effect this study is likely to produce |
| Seeds | RQ1: three seeds (1337, 1338, 1339) for **both** arms. Ablations: one seed each, read against the RQ1 spread; a delta smaller than that spread is reported as below the noise |
| Ablations | one field of `configs/grpo_base.yaml` at a time: **group size**, **KL coefficient**, and **one of** learning rate or rollouts per question |
| GRPO base | G=8, 8 questions/step, 60 steps, lr 1e-6, KL 0.04, clip 0.2, T=1.0 — ≈3840 completions ≈ 3.5 h per run at the measured 3.3 s/completion |
| Reported | per-seed accuracy, seed variability, reward curves, compute, limitations, a failure taxonomy derived from inspected examples |

Configs: [`sft_qwen05b.yaml`](../../configs/sft_qwen05b.yaml) ·
[`eval_gsm8k.yaml`](../../configs/eval_gsm8k.yaml) ·
[`eval_sft_seed1337.yaml`](../../configs/eval_sft_seed1337.yaml) ·
[`grpo_base.yaml`](../../configs/grpo_base.yaml)

## principles

- **Reward over optimizer.** A better optimizer only hacks a flawed reward faster; the scrutiny goes to the verifier.
- **Measure, don't assume.** A point estimate without seed variability is not a result. Reward improving is not accuracy improving until it has been checked.

## status

As of 2026-11-22, `master` at `59c9928`.

| Stage | Status | Evidence |
| :-- | :-- | :-- |
| Verifier | **Implemented, unit-tested** | `rewards/verifier.py`, pure functions; 37 tests pass |
| GSM8K evaluation harness | **Implemented, run** | `evaluation/`, `rlvr-eval`; 41 tests pass |
| Base-model sanity check | **Run once** | 4-shot, dev[:200]: 82/200 = 0.410, 0 no-answers. A harness check, not the RQ1 baseline |
| SFT baseline | **Implemented; validation incomplete** | `training/sft_*.py`. **Seed 1337 only, of the three RQ1 needs**: 872 steps, best dev loss 0.547 at step 400; 0-shot dev[:200]: 89/200 = 0.445 |
| Group rollout and advantages | **Not implemented** | `rollout/` is a placeholder; no tests |
| GRPO objective and loop | **Not implemented** | only the frozen config `training/grpo_config.py` and its tests |
| RQ1–RQ5 results | **None yet** | — |

The 0.410 → 0.445 move is not a result. The two numbers differ in shot count
by design (the SFT model was trained on the format), they come from one seed,
and they are not the comparison the study is about: RQ1 is SFT against GRPO,
and the GRPO arm does not exist yet.

The run artifacts above (`runs/`, `checkpoints/`) are local and git-ignored.
They are the record of what happened on one machine; they are not yet
committed evidence. See [`experiments/`](../../experiments/README.md).

## running it

```bash
make eval                                                         # base model, 4-shot, dev[:200]
uv run python -m verifiable_reward_lab.training.sft_train \
    --config configs/sft_qwen05b.yaml                             # SFT, ~1.7 h on Apple Silicon
uv run rlvr-eval --config configs/eval_sft_seed1337.yaml          # SFT checkpoint, 0-shot
```

All three download Qwen 2.5-0.5B, and the committed configs target `mps`.
