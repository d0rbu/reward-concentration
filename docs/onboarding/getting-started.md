# Getting Started

## Prerequisites

- Python 3.13
- `uv`
- a CUDA-capable GPU for later experiment workloads (the live smoke tests themselves run on CPU)

Install `uv` if needed:

```bash
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
uv sync --locked
uv run pre-commit install
uv run pytest
```

The locked environment includes the research runtime stack: PyTorch, Transformers, Datasets,
Accelerate, TRL, PEFT, safetensors, lm-eval, Matplotlib, tqdm, and Weights & Biases.

## Daily commands

```bash
uv run ruff check .
uv run ty check
uv run pytest
uv run pre-commit run --all-files
```

Run live Hugging Face integrations explicitly:

```bash
uv run pytest -m slow
```

Run safety-SFT from a TOML config, then smoke-test its held-out response perplexity:

```bash
uv run concentration sft configs/sft.toml
uv run concentration ppl configs/sft.toml outputs/safety-sft --count 16 --batch-size 4
```

The SFT command refuses to start when its output directory is already non-empty. Generated
checkpoints and run artifacts belong under ignored output paths, not in git.

Use `uv add <package>` for runtime dependencies and `uv add --dev <package>` for development-only
tooling, then commit the updated `uv.lock` with the dependency change.
