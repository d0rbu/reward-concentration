# Workflows

## Add a domain type

1. Define the predicate and phantom type in `concentration.types`.
2. Add a `parse_*` refinement for raw boundary values.
3. Register a bounded Hypothesis strategy.
4. Store only the refined type in core dataclasses and APIs.
5. Test valid generation, every invalid boundary, and serialization if applicable.

## Add tensor logic

1. Import PyTorch as `t`; do not import NumPy in `src/`.
2. Name shapes and dtypes with `jaxtyping` aliases.
3. Decorate tensor signatures with `@jaxtyped(typechecker=beartype)`.
4. Validate semantic invariants that shape typing cannot express.
5. Compare against an independent loop or NumPy-in-tests reference.
6. Add Hypothesis coverage over bounded random shapes and edge cases.

## Change an external data or model boundary

1. Inspect the real upstream artifact.
2. Encode its exact ID, schema, template, or output contract.
3. Mirror data schemas with in-memory `datasets.Dataset.from_dict(...).with_format("torch")`
   fixtures.
4. Keep default tests offline and add the real check under `slow`.
5. Update architecture, configuration, and file-reference docs.

## Add a benchmark

1. Define one end-to-end operation and explicit synchronization.
2. Run benchmarks serially, never concurrently.
3. Record the measured environment and write a new JSON file under `bench/baselines/`.
4. Commit the baseline before using it to justify an optimization.

## Run safety-SFT

1. Define `[policy]`, `[data]`, `[sft]`, and `[wandb]` tables in a TOML file.
2. Keep `max_steps` as the sole training-length representation; `epochs` is not accepted.
3. Run `uv run concentration sft <config.toml>` against a new or empty output directory.
4. Inspect `effective-config.json` and `run-manifest.json` beside the saved model and tokenizer.
5. Run `uv run concentration ppl <config.toml> <checkpoint> --count 16` for a bounded
   `heldout_probe_train` smoke metric.

SFT always rebuilds the Phase 1 prompt-disjoint splits and offset-derived spans. It does not accept
an alternate text dataset or a second string-matching mask path.

## Before handoff

```bash
uv run pre-commit run --all-files
uv run pytest -m slow
```

Report default and slow-suite results separately.
