"""Reusable single-benchmark timing and JSON-baseline helpers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One end-to-end benchmark measurement."""

    name: str
    iterations: int
    total_seconds: float
    mean_seconds: float


def benchmark(
    name: str,
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    """Warm up once, then time one non-concurrent end-to-end operation loop."""
    if not name.strip():
        raise ValueError("benchmark name must be non-empty")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    for _ in range(warmup):
        operation()
    synchronize()
    start = clock()
    for _ in range(iterations):
        operation()
    synchronize()
    total_seconds = clock() - start
    if total_seconds < 0.0:
        raise ValueError("benchmark clock must be monotonic")
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_seconds=total_seconds,
        mean_seconds=total_seconds / iterations,
    )


def write_result(path: Path, result: BenchmarkResult) -> None:
    """Write a new measured JSON baseline without overwriting an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
