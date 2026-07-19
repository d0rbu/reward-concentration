# Testing

## Default suite

```bash
uv run pytest
uv run pytest -m "not slow"
```

`not slow` is the configured default. It is deterministic, CPU-only, and provably offline: the
Hugging Face offline variables are exported in `pytest_configure` (before test modules import
libraries that snapshot them), and an autouse fixture patches `socket` so any connection attempt
in a non-slow test raises. Tests build tiny random Qwen3 models from config rather than downloading
weights. Weights & Biases is disabled. Under `CI=true` a derandomized Hypothesis profile is loaded
so CI failures reproduce locally.

Coverage measures branch coverage for `src`, reports missing lines, and fails below 95%.

## Live integration suite

```bash
uv run pytest -m slow
```

This command disables the fast coverage gate and verifies the real external contracts: both
datasets, the Qwen3 tokenizer and development policy, and the development Skywork reward model. It
requires network access and can download model weights. The smoke inputs run on CPU so the contract
does not depend on GPU visibility; experiment workloads may use CUDA.

## Pre-commit gate

```bash
uv run pre-commit run --all-files
```

The hooks run:

- `uv lock --check`;
- `uv run ruff check .`;
- `uv run ty check`;
- `uv run pytest`.

## Focused runs

```bash
uv run pytest tests/models/test_policy.py
uv run pytest tests/data/test_preference.py -k span
uv run pytest -m property
```

Keep the default suite free of network calls. Add live checks under the strict `slow` marker.
