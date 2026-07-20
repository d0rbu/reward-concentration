# Configuration

## Experiment dataclasses

All project configs are frozen and expose `from_raw` boundary parsers.

| Dataclass | Fields and defaults |
|---|---|
| `ModelConfig` | development Qwen3 policy ID, revision `main`, `bfloat16`, device `cuda` |
| `RewardModelConfig` | default 1.7B Skywork RM ID, revision `main`, `bfloat16`, device `cuda`, batch size 8, max length 2048 |
| `DataConfig` | exact preference/safety dataset IDs, max length 1024, held-out fractions 0.1/0.1, seed 0 |
| `RepExtractionConfig` | required non-negative layer and pooling enum; pooling defaults to `mean` |
| `RewardHeadConfig` | rank 1, ReLU, `linear_probe`, `diffmeans`, trainable |
| `AdvHeadConfig` | empty hidden-dimension tuple (linear) and ReLU |
| `LambdaScheduleConfig` | `dann`, `max=1.0`, `warmup_frac=0.1`, `k=10.0` |
| `KLAnchorConfig` | required reference-model ID and non-negative `gamma` |
| `ConcentrationTrainConfig` | alpha, lambda schedule, detached basis, seed, learning rates, weight decay, steps, clipping, optional alternating min-max bundle (`AlternatingMinmaxConfig` with required adversary steps), optional KL bundle |
| `TrackAConfig` | `ppo` algorithm and seed 0; `grpo` is the other accepted value |
| `WandbConfig` | `online` mode and project `reward-concentration` |
| `SFTTrainConfig` | policy and data configs, seed 0, learning rate 2e-5, per-device batch size 1, gradient accumulation 8, max steps 1000, warmup fraction 0.1, output `outputs/safety-sft`, and wandb config |

Accepted enum values are:

- pooling: `mean`, `last`, `max`, `min`;
- model dtype: `float32`, `bfloat16`;
- head nonlinearity: `relu`, `sigmoid`;
- reward-head mode: `linear_probe`, `mlp`;
- reward-head initialization: `diffmeans`, `probe`;
- lambda shape: `dann`, `linear`, `constant`, `cosine`;
- Track A: `ppo`, `grpo`;
- wandb: `online`, `offline`, `disabled`.

The concentration head/training configs are shared typed contracts only; their training
implementations are not present in the current package.

## Safety-SFT TOML

The CLI accepts exactly four optional tables and rejects unknown table or field names. Training
length has one representation, `max_steps`; an `epochs` field is invalid.

```toml
[policy]
model_id = "Qwen/Qwen3-0.6B-Base"
revision = "main"
dtype = "bfloat16"
device = "cuda"

[data]
max_len = 1024
heldout_probe_train_frac = 0.1
heldout_probe_test_frac = 0.1
seed = 0

[sft]
seed = 0
learning_rate = 2e-5
per_device_batch_size = 1
gradient_accumulation_steps = 8
max_steps = 1000
warmup_frac = 0.1
output_dir = "outputs/safety-sft"

[wandb]
mode = "online"
project = "reward-concentration"
```

The `[data]` table also accepts `preference_dataset_id` and `safety_eval_dataset_id`. `warmup_frac`
maps to `ceil(max_steps x warmup_frac)` integer warmup steps.

`uv run concentration sft <config.toml>` writes the fully defaulted configuration as JSON before
training. `uv run concentration ppl <config.toml> <checkpoint> [--count N] [--batch-size N]` uses
the policy dtype/device and data split settings from the same config.

## Package management and build

`pyproject.toml` requires Python `>=3.13,<3.14`, enables `tool.uv.package = true`, and builds
`src/concentration` with Hatchling. Runtime dependencies are:

- `torch`, `transformers`, `datasets`, `accelerate`, `trl`, `peft`;
- `safetensors`, `lm-eval`, `matplotlib`, `tqdm`, `wandb`;
- `phantom-types[hypothesis]`, `beartype`, `jaxtyping`.

Use `uv sync --locked` to reproduce `uv.lock`.

## Ruff

Ruff targets Python 3.13 and selects `E`, `F`, `W`, `I`, `UP`, `B`, `C4`, `SIM`, `RET`, and `TID`.
Line-length diagnostics remain disabled at 100 columns. First-party imports are `concentration` and
`tests`.

`flake8-tidy-imports` bans `numpy` imports. `tests/*` has a `TID251` exception so NumPy and
`hypothesis.extra.numpy` can serve as independent references.

## Pytest and coverage

Pytest collects `tests/`, enables strict config and strict markers, and defaults to `not slow`.
Configured coverage targets are:

```text
--cov=concentration
```

Coverage measures branches from `src` only and fails below 95%. Markers are `property` and `slow`.

## Type checking and pre-commit

`ty` checks Python 3.13 code. Local pre-commit hooks run lockfile validation, Ruff, ty, and the
default pytest suite. CI runs checkout, `astral-sh/setup-uv`, `uv python install 3.13`,
`uv sync --locked`, `uv run ruff check .`, `uv run ty check`, and `uv run pytest`.
