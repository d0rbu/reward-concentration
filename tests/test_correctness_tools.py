"""Executable torch examples for the template's correctness-tool stack."""

from __future__ import annotations

from typing import Any

import pytest
import torch as t
from beartype import beartype
from hypothesis import given
from hypothesis import strategies as st
from jaxtyping import Float64, TypeCheckError, jaxtyped
from phantom import Phantom

Vector = Float64[t.Tensor, "n"]


def _is_probability(value: float) -> bool:
    return 0.0 <= value <= 1.0


class Probability(float, Phantom[float], predicate=_is_probability, bound=float):
    """Example phantom type for a float in the closed interval [0, 1]."""

    @classmethod
    def __register_strategy__(cls) -> Any:
        return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def parse_probability(value: float | int | str) -> Probability:
    """Refine a raw probability at a boundary."""
    raw = float(value) if isinstance(value, int | str) else value
    return Probability.parse(raw)


@jaxtyped(typechecker=beartype)
def normalize_weights(weights: Vector) -> Vector:
    """Normalize a finite, non-negative, nonzero torch vector."""
    if weights.numel() == 0:
        raise ValueError("weights must not be empty")
    if not bool(t.isfinite(weights).all()):
        raise ValueError("weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")

    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")
    return weights / total


@jaxtyped(typechecker=beartype)
def weighted_mean(values: Vector, weights: Vector) -> Probability:
    """Return a weighted mean of probability-valued entries."""
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("values must be probabilities")
    result = float(t.dot(values, normalize_weights(weights)))
    return parse_probability(min(1.0, max(0.0, result)))


@pytest.mark.property
@given(st.from_type(Probability))
def test_phantom_type_strategy_generates_valid_probabilities(value: Probability) -> None:
    assert isinstance(value, Probability)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("raw", [0, 0.5, "0.75"])
def test_parse_probability_accepts_valid_raw_values(raw: float | int | str) -> None:
    assert isinstance(parse_probability(raw), Probability)


@pytest.mark.parametrize("raw", [-0.1, 1.1, "not-a-float"])
def test_parse_probability_rejects_invalid_raw_values(raw: float | str) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_probability(raw)


@pytest.mark.property
@given(
    st.lists(
        st.floats(min_value=1.0e-6, max_value=1.0e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    )
)
def test_normalize_weights_returns_probability_vector(raw_weights: list[float]) -> None:
    weights = t.tensor(raw_weights, dtype=t.float64)
    normalized = normalize_weights(weights)
    assert normalized.dtype == t.float64
    assert normalized.shape == weights.shape
    assert bool((normalized >= 0.0).all())
    assert float(normalized.sum()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (t.tensor([], dtype=t.float64), "empty"),
        (t.tensor([1.0, t.nan], dtype=t.float64), "finite"),
        (t.tensor([1.0, -0.5], dtype=t.float64), "non-negative"),
        (t.tensor([0.0, 0.0], dtype=t.float64), "positive"),
    ],
)
def test_normalize_weights_rejects_invalid_vectors(weights: t.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_weights(weights)


def test_weighted_mean_returns_probability() -> None:
    result = weighted_mean(
        t.tensor([0.25, 0.75], dtype=t.float64),
        t.tensor([1.0, 3.0], dtype=t.float64),
    )
    assert isinstance(result, Probability)
    assert result == pytest.approx(0.625)


def test_weighted_mean_rejects_shape_mismatch() -> None:
    with pytest.raises(TypeCheckError, match="weights"):
        weighted_mean(
            t.tensor([0.5, 0.25], dtype=t.float64),
            t.tensor([1.0], dtype=t.float64),
        )


def test_weighted_mean_rejects_non_probability_values() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        weighted_mean(
            t.tensor([0.5, 1.5], dtype=t.float64),
            t.tensor([1.0, 1.0], dtype=t.float64),
        )
