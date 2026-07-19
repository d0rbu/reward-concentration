# File Reference

## Top level

| File | Purpose |
|---|---|
| `README.md` | Project scope, verified external IDs, setup, and test commands |
| `AGENTS.md` | Agent entry point and binding engineering conventions |
| `CLAUDE.md` | Pointer to shared agent conventions |
| `pyproject.toml` | Package metadata, dependencies, build system, and tool configuration |
| `uv.lock` | Locked Python dependency graph |
| `.pre-commit-config.yaml` | Local lock, lint, type, and test hooks |
| `.python-version` | Python version selected by local tooling |
| `.gitignore` | Generated artifacts and private local planning files excluded from git |
| `LICENSE` | MIT license |

## Package

| File | Purpose |
|---|---|
| `src/concentration/__init__.py` | Package marker and summary |
| `src/concentration/types.py` | Phantom scalars, parsers, tensor aliases, and validating smart constructors |
| `src/concentration/config.py` | Frozen configs, enums, defaults, and boundary parsing |
| `src/concentration/seeding.py` | Python, torch CPU/CUDA, and Transformers seeding |
| `src/concentration/run_logging.py` | JSONL-always logger with optional wandb mirroring |
| `src/concentration/benchmark.py` | Synchronized benchmark timing and JSON result writing |
| `src/concentration/data/__init__.py` | Data-package marker |
| `src/concentration/data/preference.py` | Preference schema, prompt splits, deduplication, chat spans, and length filtering |
| `src/concentration/data/safety_eval.py` | BeaverTails evaluation schema and prompt loading |
| `src/concentration/models/__init__.py` | Model-package marker |
| `src/concentration/models/policy.py` | Causal-LM loading, exact layer hook, and four pooling modes |
| `src/concentration/models/reward_model.py` | Reward protocol, sequence-classification adapter, raw-score cache, and diagnostics |
| `src/concentration/eval/__init__.py` | Evaluation-package marker |
| `src/concentration/eval/stats.py` | Torch-only confidence interval, Welch t, R-squared, and AUC |

## Tests

| File | Purpose |
|---|---|
| `tests/conftest.py` | Deterministic, wandb-disabled, offline-fast test environment |
| `tests/test_correctness_tools.py` | Torch scaffold examples for phantom, runtime, tensor, and property checks |
| `tests/test_types.py` | Scalar parser and tensor-wrapper unit/property tests |
| `tests/test_config.py` | Frozen dataclass, enum, parser, and all-or-nothing bundle tests |
| `tests/test_seeding.py` | Cross-backend seed and reproducibility tests |
| `tests/test_run_logging.py` | JSONL/wandb routing and invalid-record tests |
| `tests/test_benchmark_runner.py` | Benchmark timing, validation, and JSON serialization tests |
| `tests/data/test_preference.py` | In-memory schema/split/span/filter tests and live PKU/Qwen tokenizer checks |
| `tests/data/test_safety_eval.py` | In-memory schema/content tests and live BeaverTails check |
| `tests/models/test_policy.py` | Tiny-Qwen exact hook and pooling tests plus live development-policy check |
| `tests/models/test_reward_model.py` | Tiny reward scoring/cache tests plus live Skywork check |
| `tests/eval/test_stats.py` | Hand, loop, NumPy, and property references for statistics |

## Benchmarks

| File or path | Purpose |
|---|---|
| `bench/__init__.py` | Benchmark-directory marker |
| `bench/run_benchmark.py` | Thin import surface for phase-specific benchmark scripts |
| `bench/baselines/` | Checked-in measured JSON baselines; intentionally empty in shared infrastructure |

## Documentation

| Path | Purpose |
|---|---|
| `docs/README.md` | Documentation index |
| `docs/onboarding/getting-started.md` | Setup and daily commands |
| `docs/onboarding/workflows.md` | Type, tensor, external-boundary, and benchmark workflows |
| `docs/onboarding/glossary.md` | Project vocabulary |
| `docs/development/correctness.md` | Scalar, tensor, runtime, and property-test standards |
| `docs/development/testing.md` | Default, slow, focused, coverage, and pre-commit test behavior |
| `docs/pipelines/experiment-lifecycle.md` | Lifecycle for experiments using shared infrastructure |
| `docs/reference/architecture.md` | Module behavior and exact external data contracts |
| `docs/reference/configuration.md` | Dataclass defaults, dependency, lint, test, and type configuration |
| `docs/reference/file-reference.md` | This complete tracked-file map |

## CI

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Runs Ruff, ty, and the default offline pytest suite on pushes and pull requests |
