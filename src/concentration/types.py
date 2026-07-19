"""Validated scalar and tensor domain types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch as t
from beartype import beartype
from hypothesis import strategies as st
from jaxtyping import Float, Float32, jaxtyped
from phantom import Phantom

ORTHONORMAL_ATOL = 1.0e-5
MAX_SEED = 2**32 - 1
OrthonormalTensor = Float32[t.Tensor, "hidden rank"]
FloatMatrix = Float[t.Tensor, "hidden rank"]
PooledTensor = Float32[t.Tensor, "batch hidden"]
FloatBatchMatrix = Float[t.Tensor, "batch hidden"]
ScoreTensor = Float32[t.Tensor, "batch"]
FloatScoreTensor = Float[t.Tensor, "batch"]


def _is_rank(value: int) -> bool:
    return not isinstance(value, bool) and value > 0


def _is_seed(value: int) -> bool:
    return not isinstance(value, bool) and 0 <= value <= MAX_SEED


def _is_unit_interval(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


def _is_non_negative_float(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


class Rank(int, Phantom[int], predicate=_is_rank, bound=int):
    """A strictly positive integer rank or count."""

    @classmethod
    def __register_strategy__(cls) -> Any:
        return st.integers(min_value=1, max_value=64)


class Seed(int, Phantom[int], predicate=_is_seed, bound=int):
    """A seed accepted by all project seeding backends."""

    @classmethod
    def __register_strategy__(cls) -> Any:
        return st.integers(min_value=0, max_value=MAX_SEED)


class UnitInterval(float, Phantom[float], predicate=_is_unit_interval, bound=float):
    """A finite float in the closed interval [0, 1]."""

    @classmethod
    def __register_strategy__(cls) -> Any:
        return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


class NonNegativeFloat(
    float,
    Phantom[float],
    predicate=_is_non_negative_float,
    bound=float,
):
    """A finite, non-negative scalar used for loss weights and rates."""

    @classmethod
    def __register_strategy__(cls) -> Any:
        return st.floats(
            min_value=0.0,
            max_value=1.0e6,
            allow_nan=False,
            allow_infinity=False,
        )


@beartype
def parse_rank(value: int | str) -> Rank:
    """Refine an integer-like boundary value into a positive rank."""
    if isinstance(value, bool):
        raise TypeError("rank must be an integer, not bool")
    raw = int(value) if isinstance(value, str) else value
    return Rank.parse(raw)


@beartype
def parse_seed(value: int | str) -> Seed:
    """Refine an integer-like boundary value into a portable seed."""
    if isinstance(value, bool):
        raise TypeError("seed must be an integer, not bool")
    raw = int(value) if isinstance(value, str) else value
    return Seed.parse(raw)


@beartype
def parse_unit_interval(value: float | int | str) -> UnitInterval:
    """Refine a numeric boundary value into a unit-interval float."""
    if isinstance(value, bool):
        raise TypeError("unit-interval value must be numeric, not bool")
    raw = float(value) if isinstance(value, int | str) else value
    return UnitInterval.parse(raw)


@beartype
def parse_non_negative_float(value: float | int | str) -> NonNegativeFloat:
    """Refine a numeric boundary value into a finite non-negative float."""
    if isinstance(value, bool):
        raise TypeError("non-negative value must be numeric, not bool")
    raw = float(value) if isinstance(value, int | str) else value
    return NonNegativeFloat.parse(raw)


@dataclass(frozen=True, slots=True)
class OrthonormalMatrix:
    """A non-empty fp32 matrix whose columns are orthonormal."""

    tensor: OrthonormalTensor

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_tensor(
        cls, value: FloatMatrix
    ) -> OrthonormalMatrix:
        """Validate and wrap an orthonormal matrix without copying it."""
        if value.dtype != t.float32:
            raise TypeError("orthonormal matrices must use torch.float32")
        hidden, rank = value.shape
        if hidden == 0 or rank == 0:
            raise ValueError("orthonormal matrices must be non-empty")
        if rank > hidden:
            raise ValueError("orthonormal matrices cannot have more columns than rows")
        if not bool(t.isfinite(value).all()):
            raise ValueError("orthonormal matrices must be finite")
        identity = t.eye(rank, dtype=value.dtype, device=value.device)
        error = (value.mT @ value - identity).abs().max()
        if float(error) > ORTHONORMAL_ATOL:
            raise ValueError(
                f"columns are not orthonormal: max error {float(error):.8g} "
                f"> {ORTHONORMAL_ATOL}"
            )
        return cls(value)


@dataclass(frozen=True, slots=True)
class PooledRepresentations:
    """A non-empty batch of finite fp32 pooled representations."""

    tensor: PooledTensor

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_tensor(
        cls, value: FloatBatchMatrix
    ) -> PooledRepresentations:
        """Validate and wrap pooled representations without copying them."""
        if value.dtype != t.float32:
            raise TypeError("pooled representations must use torch.float32")
        batch, hidden = value.shape
        if batch == 0 or hidden == 0:
            raise ValueError("pooled representations must be non-empty")
        if not bool(t.isfinite(value).all()):
            raise ValueError("pooled representations must be finite")
        return cls(value)


@dataclass(frozen=True, slots=True)
class ScoreBatch:
    """A non-empty batch of finite raw reward-model scores."""

    tensor: ScoreTensor

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_tensor(cls, value: FloatScoreTensor) -> ScoreBatch:
        """Validate and wrap raw reward scores without copying them."""
        if value.dtype != t.float32:
            raise TypeError("score batches must use torch.float32")
        if value.shape[0] == 0:
            raise ValueError("score batches must be non-empty")
        if not bool(t.isfinite(value).all()):
            raise ValueError("score batches must be finite")
        return cls(value)
