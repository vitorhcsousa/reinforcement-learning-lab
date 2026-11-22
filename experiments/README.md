# Experiments

The committed record of what was run. **Empty so far**: no experiment has
been committed yet.

## what goes here

One directory per hypothesis test, for any study:

```text
experiments/<study>/<slug>/
├── README.md                     # question, protocol, result, deviations
└── <arm>/seed-<n>/
    ├── config.yaml               # the run that actually happened, overrides included
    ├── metrics.jsonl
    └── summary.json
```

`metrics.jsonl` and `summary.json` are the record; plots and wandb are views
of it. Never checkpoints, weights, datasets or raw logs. Those stay local.

## what does not

| Path | What it is | Tracked |
| :-- | :-- | :-- |
| `runs/` | output of `rlvr-train`, `rlvr-eval` and the SFT trainer on one machine | no — git-ignored |
| `checkpoints/` | model weights | no — git-ignored |
| `data/` | corpora, fetched and checksum-verified by `make data` | no — git-ignored |

The tiny reference run (`configs/tiny.yaml`) is a reproducibility check, not
an experiment. Its number is quoted in the root README and recomputed by
`make reference`, so it does not live here.

A result from `runs/` becomes evidence when its config, metrics and summary
are copied here with a README that says what was asked and what was found.
