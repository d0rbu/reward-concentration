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

## Before handoff

```bash
uv run pre-commit run --all-files
uv run pytest -m slow
```

Report default and slow-suite results separately.
