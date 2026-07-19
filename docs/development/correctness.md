# Correctness

The project bias is to make bad state unrepresentable and to crash at the boundary when that is not
possible.

## Scalar refinement

Use `phantom-types` for domain concepts narrower than Python primitives. The project defines:

- `Rank`: positive integer;
- `Seed`: integer in `[0, 2**32 - 1]`;
- `UnitInterval`: finite float in `[0, 1]`;
- `NonNegativeFloat`: finite float greater than or equal to zero.

Raw strings and numbers enter through the matching `parse_*` functions. Frozen configuration
dataclasses store the refined values.

## Tensor contracts

PyTorch is imported as `t`. NumPy imports are banned in `src/` and permitted in tests only for
independent reference calculations.

Every tensor signature combines `jaxtyping` with `beartype`:

```python
import torch as t
from beartype import beartype
from jaxtyping import Float, jaxtyped

Batch = Float[t.Tensor, "batch hidden"]

@jaxtyped(typechecker=beartype)
def transform(values: Batch) -> Batch:
    return values
```

Semantic tensor invariants use smart constructors. `OrthonormalMatrix` requires non-empty finite
fp32 values and `max(abs(Q.T @ Q - I)) <= 1e-5`; `PooledRepresentations` and `ScoreBatch` require
finite, non-empty fp32 batches.

## Runtime boundaries

Validate schema, dtype, shape, finite values, span order, cache completeness, and model-output
contracts before downstream code can consume them. Invalid input raises immediately. Do not catch
exceptions to substitute defaults or silently truncate data.

## Property and reference tests

Use Hypothesis for random shapes, padding patterns, scalar domains, and algebraic invariants.
Novel numeric logic also needs an independently written reference: Python loops or NumPy in tests
are preferred when they do not share the implementation's vectorized codepath.
