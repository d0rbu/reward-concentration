"""Small-sample statistics implemented with torch only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import torch as t
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped

Sample = Float[t.Tensor, "sample"]
FirstSample = Float[t.Tensor, "first_sample"]
SecondSample = Float[t.Tensor, "second_sample"]
BinaryLabels = Int[t.Tensor, "sample"] | Bool[t.Tensor, "sample"]

# Two-sided 95% Student-t critical values from the NIST/SEMATECH e-Handbook,
# section 1.3.6.7.2, "Critical Values of the Student's t Distribution".
_T_CRITICAL_95 = {
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
}
@dataclass(frozen=True, slots=True)
class MeanConfidenceInterval:
    """Mean and symmetric two-sided 95% confidence interval."""

    mean: float
    lower: float
    upper: float
    half_width: float
    sample_size: int


@dataclass(frozen=True, slots=True)
class WelchTest:
    """Welch t statistic and floor-rounded Welch-Satterthwaite degrees of freedom."""

    t_statistic: float
    degrees_of_freedom: int


@jaxtyped(typechecker=beartype)
def _require_finite(values: Sample, name: str) -> None:
    if not bool(t.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")


def _critical_value_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 2:
        raise ValueError("95% confidence intervals require at least 3 samples")
    if degrees_of_freedom not in _T_CRITICAL_95:
        raise ValueError(
            "95% confidence intervals support 3-10 samples (sourced t table, df 2-9); "
            f"got df={degrees_of_freedom}. Extend the table with sourced values if needed."
        )
    return _T_CRITICAL_95[degrees_of_freedom]


@jaxtyped(typechecker=beartype)
def mean_confidence_interval(
    values: Sample,
) -> MeanConfidenceInterval:
    """Compute a mean and two-sided 95% CI using the small-sample t table."""
    _require_finite(values, "values")
    sample_size = values.shape[0]
    critical = _critical_value_95(sample_size - 1)
    mean = float(values.mean())
    standard_error = float(values.std(unbiased=True)) / math.sqrt(sample_size)
    half_width = critical * standard_error
    return MeanConfidenceInterval(
        mean=mean,
        lower=mean - half_width,
        upper=mean + half_width,
        half_width=half_width,
        sample_size=sample_size,
    )


def _exact_variance_term(values: t.Tensor) -> Fraction:
    """Unbiased sample variance over sample size, in exact rational arithmetic.

    Float inputs convert to Fractions exactly, so the Welch-Satterthwaite df and its
    floor are mathematically exact — float rounding cannot cross an integer boundary.
    """
    fractions = [Fraction(value) for value in values.tolist()]
    size = len(fractions)
    mean = sum(fractions, start=Fraction(0)) / size
    squared_deviations = sum(
        ((value - mean) ** 2 for value in fractions), start=Fraction(0)
    )
    return squared_deviations / (size - 1) / size


@jaxtyped(typechecker=beartype)
def welch_t_test(
    first: FirstSample,
    second: SecondSample,
) -> WelchTest:
    """Compute Welch's t statistic and the exact floor of the Welch-Satterthwaite df."""
    _require_finite(first, "first")
    _require_finite(second, "second")
    first_size = first.shape[0]
    second_size = second.shape[0]
    if first_size < 2 or second_size < 2:
        raise ValueError("Welch's t test requires at least two values per sample")

    first_term = _exact_variance_term(first)
    second_term = _exact_variance_term(second)
    standard_error_squared = first_term + second_term
    if standard_error_squared == 0:
        raise ValueError("Welch's t statistic is undefined when both samples are constant")

    statistic = (float(first.mean()) - float(second.mean())) / math.sqrt(
        float(standard_error_squared)
    )
    denominator = first_term**2 / (first_size - 1) + second_term**2 / (second_size - 1)
    degrees_of_freedom = math.floor(standard_error_squared**2 / denominator)
    return WelchTest(t_statistic=statistic, degrees_of_freedom=degrees_of_freedom)


@jaxtyped(typechecker=beartype)
def r_squared(
    targets: Sample,
    predictions: Sample,
) -> float:
    """Compute the coefficient of determination against the target mean."""
    _require_finite(targets, "targets")
    _require_finite(predictions, "predictions")
    if targets.shape[0] == 0:
        raise ValueError("R-squared requires a non-empty sample")
    total_sum_squares = ((targets - targets.mean()) ** 2).sum()
    if float(total_sum_squares) == 0.0:
        raise ValueError("R-squared is undefined for constant targets")
    residual_sum_squares = ((targets - predictions) ** 2).sum()
    return 1.0 - float(residual_sum_squares / total_sum_squares)


@jaxtyped(typechecker=beartype)
def binary_auc(
    labels: BinaryLabels,
    scores: Sample,
) -> float:
    """Compute binary ROC AUC as pairwise ranking probability, with half credit for ties."""
    _require_finite(scores, "scores")
    if labels.shape[0] == 0:
        raise ValueError("AUC requires a non-empty sample")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("AUC labels must be binary")
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("AUC requires both positive and negative labels")
    comparisons = positives[:, None] - negatives[None, :]
    wins = (comparisons > 0).to(scores.dtype)
    ties = (comparisons == 0).to(scores.dtype)
    return float((wins + 0.5 * ties).mean())
