# Reward Concentration

Correctness-first infrastructure for research on preference representation concentration in
language models. The current package establishes the shared data, model, caching, statistics,
configuration, seeding, logging, and benchmark boundaries needed by later experiments; it does
not yet implement concentration training or report research results. It now produces the
safety-SFT checkpoint that later concentration experiments start from.

## Current scope

- validated phantom scalar types and fp32 tensor wrappers
- frozen, boundary-parsed experiment configuration
- schema-locked PKU preference and BeaverTails evaluation loaders
- deterministic prompt-level preference splits, deduplication, chat spans, and length filtering
- Qwen3 causal-LM loading, exact layer hooks, and four response-pooling modes
- frozen Skywork sequence-classification reward scoring and verified raw-score caches
- torch-only confidence intervals, Welch t tests, R-squared, and binary AUC
- deterministic seeding plus JSONL-always and optional Weights & Biases logging
- TRL safety-SFT on train-split preferred responses with exact response-token loss masking
- held-out response token-mean NLL and perplexity for before/after SFT smoke evaluation
- a benchmark runner and an intentionally empty measured-baseline directory

Raw reward-model logits are preserved exactly. Score means and standard deviations are diagnostics
only and are never used to standardize targets.

## Verified external contracts

| Role | Hugging Face ID |
|---|---|
| Development policy | `Qwen/Qwen3-0.6B-Base` |
| Main policy | `Qwen/Qwen3-1.7B-Base` |
| Development reward model | `Skywork/Skywork-Reward-V2-Qwen3-0.6B` |
| Default reward model | `Skywork/Skywork-Reward-V2-Qwen3-1.7B` |
| Preference data | `PKU-Alignment/PKU-SafeRLHF-single-dimension` |
| Safety-evaluation prompts | `PKU-Alignment/BeaverTails-Evaluation` |

The loaders assert the observed schemas before returning data. The development-tier models and
both datasets are exercised live by the slow suite; the main-tier IDs are recorded defaults that
have not been live-tested yet. See
[`docs/reference/architecture.md`](docs/reference/architecture.md) for the exact fields.

## Setup and checks

```bash
uv sync
uv run pre-commit install
uv run pre-commit run --all-files
```

The default test suite is deterministic, CPU-only, and provably offline (socket-blocked):

```bash
uv run pytest
uv run pytest -m "not slow"
```

Live dataset and model checks are separate, require network access, and can download model weights:

```bash
uv run pytest -m slow
```

## Repository layout

```text
src/concentration/  importable research infrastructure
tests/              deterministic unit, property, integration, and slow tests
bench/              measured JSON baseline directory (the runner lives in the package)
docs/               source-of-truth project documentation
.github/workflows/  continuous integration checks
```

Start with [`docs/README.md`](docs/README.md). The project is MIT licensed; see
[`LICENSE`](LICENSE).
