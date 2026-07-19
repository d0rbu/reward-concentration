from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch as t
from hypothesis import given
from hypothesis import strategies as st
from jaxtyping import TypeCheckError

from concentration.types import (
    MAX_SEED,
    NonNegativeFloat,
    OrthonormalMatrix,
    PooledRepresentations,
    Rank,
    ScoreBatch,
    Seed,
    UnitInterval,
    parse_non_negative_float,
    parse_rank,
    parse_seed,
    parse_unit_interval,
)


@pytest.mark.property
@given(
    st.from_type(Rank),
    st.from_type(Seed),
    st.from_type(UnitInterval),
    st.from_type(NonNegativeFloat),
)
def test_registered_phantom_strategies_respect_domains(
    rank: Rank,
    seed: Seed,
    unit: UnitInterval,
    non_negative: NonNegativeFloat,
) -> None:
    assert rank > 0
    assert 0 <= seed <= MAX_SEED
    assert 0.0 <= unit <= 1.0
    assert non_negative >= 0.0


@pytest.mark.parametrize(
    ("parser", "raw", "expected_type", "expected"),
    [
        (parse_rank, "3", Rank, 3),
        (parse_seed, "0", Seed, 0),
        (parse_unit_interval, "0.25", UnitInterval, 0.25),
        (parse_non_negative_float, 2, NonNegativeFloat, 2.0),
    ],
)
def test_scalar_parsers_refine_valid_boundary_values(
    parser: Callable[[Any], object],
    raw: object,
    expected_type: type[object],
    expected: float,
) -> None:
    parsed = parser(raw)
    assert isinstance(parsed, expected_type)
    assert parsed == expected


@pytest.mark.parametrize(
    ("parser", "raw"),
    [
        (parse_rank, 0),
        (parse_rank, -1),
        (parse_rank, True),
        (parse_rank, "1.5"),
        (parse_seed, -1),
        (parse_seed, MAX_SEED + 1),
        (parse_seed, False),
        (parse_unit_interval, -0.01),
        (parse_unit_interval, 1.01),
        (parse_unit_interval, float("nan")),
        (parse_unit_interval, True),
        (parse_non_negative_float, -0.01),
        (parse_non_negative_float, float("inf")),
        (parse_non_negative_float, False),
    ],
)
def test_scalar_parsers_crash_on_invalid_values(
    parser: Callable[[Any], object],
    raw: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        parser(raw)


@pytest.mark.property
@given(
    hidden=st.integers(min_value=1, max_value=12),
    rank=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_orthonormal_matrix_accepts_qr_outputs(hidden: int, rank: int, seed: int) -> None:
    hidden = max(hidden, rank)
    generator = t.Generator().manual_seed(seed)
    matrix = t.randn(hidden, rank, generator=generator, dtype=t.float32)
    q, _ = t.linalg.qr(matrix, mode="reduced")
    wrapped = OrthonormalMatrix.from_tensor(q)
    assert wrapped.tensor is q


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    [
        (t.eye(2, dtype=t.float64), TypeError, "float32"),
        (t.empty((0, 0), dtype=t.float32), ValueError, "non-empty"),
        (t.ones((2, 3), dtype=t.float32), ValueError, "columns"),
        (t.tensor([[1.0, 1.0], [0.0, 0.0]]), ValueError, "not orthonormal"),
        (t.tensor([[float("nan")]], dtype=t.float32), ValueError, "finite"),
    ],
)
def test_orthonormal_matrix_rejects_invalid_values(
    matrix: t.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        OrthonormalMatrix.from_tensor(matrix)


def test_orthonormal_matrix_rejects_wrong_rank() -> None:
    with pytest.raises(TypeCheckError):
        OrthonormalMatrix.from_tensor(t.ones(2, dtype=t.float32))


@pytest.mark.property
@given(
    batch=st.integers(min_value=1, max_value=8),
    hidden=st.integers(min_value=1, max_value=16),
)
def test_pooled_representations_accept_finite_nonempty_batches(batch: int, hidden: int) -> None:
    values = t.zeros((batch, hidden), dtype=t.float32)
    assert PooledRepresentations.from_tensor(values).tensor is values


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (t.ones((1, 2), dtype=t.float64), TypeError, "float32"),
        (t.empty((0, 2), dtype=t.float32), ValueError, "non-empty"),
        (t.empty((2, 0), dtype=t.float32), ValueError, "non-empty"),
        (t.tensor([[float("inf")]], dtype=t.float32), ValueError, "finite"),
    ],
)
def test_pooled_representations_reject_invalid_batches(
    values: t.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        PooledRepresentations.from_tensor(values)


@pytest.mark.property
@given(values=st.lists(st.floats(-100, 100), min_size=1, max_size=16))
def test_score_batch_accepts_finite_nonempty_fp32(values: list[float]) -> None:
    scores = t.tensor(values, dtype=t.float32)
    assert ScoreBatch.from_tensor(scores).tensor is scores


@pytest.mark.parametrize(
    ("scores", "error", "message"),
    [
        (t.ones(2, dtype=t.float64), TypeError, "float32"),
        (t.empty(0, dtype=t.float32), ValueError, "non-empty"),
        (t.tensor([float("nan")], dtype=t.float32), ValueError, "finite"),
    ],
)
def test_score_batch_rejects_invalid_batches(
    scores: t.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ScoreBatch.from_tensor(scores)


def test_batch_wrappers_reject_wrong_tensor_ranks() -> None:
    with pytest.raises(TypeCheckError):
        PooledRepresentations.from_tensor(t.ones(2, dtype=t.float32))
    with pytest.raises(TypeCheckError):
        ScoreBatch.from_tensor(t.ones((1, 2), dtype=t.float32))


def test_orthonormal_tolerance_boundary_is_pinned() -> None:
    accepted = t.diag(t.tensor([1.0, 1.0 + 2.0e-6], dtype=t.float32))
    OrthonormalMatrix.from_tensor(accepted)
    rejected = t.diag(t.tensor([1.0, 1.0 + 2.6e-5], dtype=t.float32))
    with pytest.raises(ValueError, match="not orthonormal"):
        OrthonormalMatrix.from_tensor(rejected)
