from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pytest
import torch as t
from hypothesis import given
from hypothesis import strategies as st
from jaxtyping import TypeCheckError

from concentration.eval.stats import (
    binary_auc,
    mean_confidence_interval,
    r_squared,
    welch_t_test,
)


def _exact_variance_term(values: list[float]) -> Fraction:
    fractions = [Fraction.from_float(value) for value in values]
    mean = sum(fractions, start=Fraction(0)) / Fraction(len(fractions))
    squared_deviations = sum(
        ((value - mean) ** 2 for value in fractions),
        start=Fraction(0),
    )
    variance = squared_deviations / Fraction(len(fractions) - 1)
    return variance / Fraction(len(fractions))


def test_mean_confidence_interval_uses_small_sample_t_value() -> None:
    result = mean_confidence_interval(t.tensor([1.0, 2.0, 3.0], dtype=t.float64))
    expected_half_width = 4.303 / math.sqrt(3.0)
    assert result.mean == pytest.approx(2.0)
    assert result.half_width == pytest.approx(expected_half_width)
    assert result.lower == pytest.approx(2.0 - expected_half_width)
    assert result.upper == pytest.approx(2.0 + expected_half_width)
    assert result.sample_size == 3


@pytest.mark.property
@given(
    values=st.lists(
        st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=10,
    )
)
def test_mean_confidence_interval_matches_numpy_reference(values: list[float]) -> None:
    array = np.asarray(values, dtype=np.float64)
    result = mean_confidence_interval(t.tensor(values, dtype=t.float64))
    critical = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}[
        len(values) - 1
    ]
    expected = critical * float(array.std(ddof=1)) / math.sqrt(len(values))
    assert result.mean == pytest.approx(float(array.mean()))
    assert result.half_width == pytest.approx(expected)


def test_mean_confidence_interval_crashes_beyond_sourced_table() -> None:
    with pytest.raises(ValueError, match="3-10 samples"):
        mean_confidence_interval(t.arange(11, dtype=t.float64))


@pytest.mark.parametrize(
    "values",
    [
        t.tensor([1.0, 2.0], dtype=t.float32),
        t.tensor([1.0, float("nan"), 2.0], dtype=t.float32),
    ],
)
def test_mean_confidence_interval_rejects_invalid_samples(values: t.Tensor) -> None:
    with pytest.raises(ValueError):
        mean_confidence_interval(values)


@pytest.mark.property
@given(
    first=st.lists(
        st.floats(-20, 20, allow_subnormal=False, width=32),
        min_size=2,
        max_size=12,
    ),
    second=st.lists(
        st.floats(-20, 20, allow_subnormal=False, width=32),
        min_size=2,
        max_size=12,
    ),
)
def test_welch_t_matches_independent_numpy_formula(
    first: list[float],
    second: list[float],
) -> None:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    first_term = float(first_array.var(ddof=1)) / len(first)
    second_term = float(second_array.var(ddof=1)) / len(second)
    if first_term + second_term == 0.0:
        with pytest.raises(ValueError):
            welch_t_test(t.tensor(first), t.tensor(second))
        return
    expected_t = (float(first_array.mean()) - float(second_array.mean())) / math.sqrt(
        first_term + second_term
    )
    denominator = first_term**2 / (len(first) - 1) + second_term**2 / (len(second) - 1)
    if denominator == 0.0:
        with pytest.raises(ValueError):
            welch_t_test(t.tensor(first), t.tensor(second))
        return
    exact_first_term = _exact_variance_term(first)
    exact_second_term = _exact_variance_term(second)
    exact_denominator = exact_first_term**2 / (len(first) - 1) + exact_second_term**2 / (
        len(second) - 1
    )
    expected_df = math.floor(
        (exact_first_term + exact_second_term) ** 2 / exact_denominator
    )
    result = welch_t_test(
        t.tensor(first, dtype=t.float64),
        t.tensor(second, dtype=t.float64),
    )
    assert result.t_statistic == pytest.approx(expected_t)
    assert result.degrees_of_freedom == expected_df


def test_welch_df_does_not_floor_one_ulp_below_an_exact_integer() -> None:
    result = welch_t_test(
        t.tensor([0.0, 0.0], dtype=t.float64),
        t.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 7.0, -1.4509320259094238], dtype=t.float64),
    )
    assert result.degrees_of_freedom == 7


def test_welch_t_rejects_small_constant_and_nonfinite_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        welch_t_test(t.tensor([1.0]), t.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="constant"):
        welch_t_test(t.ones(3), t.ones(4))
    with pytest.raises(ValueError, match="finite"):
        welch_t_test(t.tensor([1.0, float("inf")]), t.tensor([1.0, 2.0]))


@pytest.mark.property
@given(
    targets=st.lists(st.floats(-20, 20), min_size=2, max_size=20),
    offsets=st.lists(st.floats(-5, 5), min_size=2, max_size=20),
)
def test_r_squared_matches_numpy_reference(targets: list[float], offsets: list[float]) -> None:
    size = min(len(targets), len(offsets))
    target_array = np.asarray(targets[:size], dtype=np.float64)
    prediction_array = target_array + np.asarray(offsets[:size], dtype=np.float64)
    denominator = float(((target_array - target_array.mean()) ** 2).sum())
    if denominator == 0.0:
        with pytest.raises(ValueError):
            r_squared(t.tensor(target_array), t.tensor(prediction_array))
        return
    expected = 1.0 - float(((target_array - prediction_array) ** 2).sum()) / denominator
    assert r_squared(
        t.tensor(target_array, dtype=t.float64),
        t.tensor(prediction_array, dtype=t.float64),
    ) == pytest.approx(expected)


def test_r_squared_known_cases_and_validation() -> None:
    targets = t.tensor([1.0, 2.0, 3.0])
    assert r_squared(targets, targets) == 1.0
    assert r_squared(targets, t.full_like(targets, 2.0)) == 0.0
    with pytest.raises(ValueError, match="constant"):
        r_squared(t.ones(3), t.zeros(3))
    with pytest.raises(ValueError, match="non-empty"):
        r_squared(t.empty(0), t.empty(0))
    with pytest.raises(TypeCheckError):
        r_squared(t.ones(2), t.ones(3))


def _loop_auc(labels: Sequence[int], scores: Sequence[int | float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    total = 0.0
    for positive in positives:
        for negative in negatives:
            total += float(positive > negative) + 0.5 * float(positive == negative)
    return total / (len(positives) * len(negatives))


@pytest.mark.property
@given(
    positive_scores=st.lists(st.integers(-5, 5), min_size=1, max_size=10),
    negative_scores=st.lists(st.integers(-5, 5), min_size=1, max_size=10),
)
def test_binary_auc_matches_pairwise_loop_reference(
    positive_scores: list[int],
    negative_scores: list[int],
) -> None:
    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    scores = [*positive_scores, *negative_scores]
    expected = _loop_auc(labels, scores)
    actual = binary_auc(t.tensor(labels), t.tensor(scores, dtype=t.float32))
    assert actual == pytest.approx(expected)


def test_binary_auc_known_ties_and_validation() -> None:
    assert binary_auc(t.tensor([1, 0]), t.tensor([2.0, 1.0])) == 1.0
    assert binary_auc(t.tensor([True, False]), t.tensor([1.0, 1.0])) == 0.5
    with pytest.raises(ValueError, match="binary"):
        binary_auc(t.tensor([0, 2]), t.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="both"):
        binary_auc(t.tensor([1, 1]), t.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="non-empty"):
        binary_auc(t.empty(0, dtype=t.int64), t.empty(0))
    with pytest.raises(ValueError, match="finite"):
        binary_auc(t.tensor([0, 1]), t.tensor([0.0, float("nan")]))
