# AGENTS.md — reward-concentration

This is a correctness-first Python ML research codebase. Keep the package explicit, fail loudly at
invalid boundaries, and keep code, tests, and documentation synchronized.

## Read these first

- [`README.md`](README.md) — project scope and quickstart
- [`docs/README.md`](docs/README.md) — documentation map
- [`docs/onboarding/getting-started.md`](docs/onboarding/getting-started.md) — local setup
- [`docs/onboarding/workflows.md`](docs/onboarding/workflows.md) — common development flows
- [`docs/development/correctness.md`](docs/development/correctness.md) — correctness philosophy
- [`docs/reference/architecture.md`](docs/reference/architecture.md) — package architecture
- [`docs/reference/configuration.md`](docs/reference/configuration.md) — code and tool configuration
- [`docs/reference/file-reference.md`](docs/reference/file-reference.md) — file-by-file reference

## Repository layout

```text
src/concentration/  package code
tests/              unit, property, integration, and slow tests
bench/              benchmark entry surface and JSON baselines
docs/               source-of-truth documentation
.github/workflows/  CI checks
```

## Conventions

- Use `uv sync` to install and `uv run ...` to invoke tools.
- Run `uv run pre-commit run --all-files` before handoff.
- Import PyTorch as `import torch as t` throughout `src/`.
- Never import NumPy in `src/`; Ruff enforces this. NumPy is allowed in tests as an independent
  numeric reference.
- Put `jaxtyping` shapes/dtypes plus `@jaxtyped(typechecker=beartype)` on tensor signatures.
- Refine raw scalar values into phantom types at boundaries.
- Prefer making bad state unrepresentable; otherwise validate and crash immediately.
- Do not add defensive fallbacks or exception swallowing.
- Use Hypothesis for broad invariants and independent references for novel numeric logic.
- Keep imports at the top of each file.
- Update `docs/reference/file-reference.md` whenever files change.

## Correctness tools

The scaffold demonstration in `tests/test_correctness_tools.py` now uses torch and preserves the
template's examples of:

- a closed-range probability phantom type and `parse_probability` refinement;
- `jaxtyping` plus `beartype` tensor contracts;
- property tests generated from phantom types.

Project-owned types live in `src/concentration/types.py`: positive `Rank`, bounded `Seed`,
`UnitInterval`, `NonNegativeFloat`, `OrthonormalMatrix`, `PooledRepresentations`, and `ScoreBatch`.

## Testing

```bash
uv run pre-commit run --all-files
uv run pytest -m slow  # separate live-network/model integration suite
```

The default suite excludes `slow`, runs without network or CUDA, and enforces 95% branch coverage
across both `src` and `tests`.
