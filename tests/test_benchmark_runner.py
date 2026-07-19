from __future__ import annotations

import json
from pathlib import Path

import pytest

from concentration.benchmark import BenchmarkResult, benchmark, write_result


def test_benchmark_runs_warmup_and_measured_iterations_with_sync() -> None:
    calls: list[str] = []
    times = iter([10.0, 14.0])

    def clock() -> float:
        calls.append("clock")
        return next(times)

    result = benchmark(
        "step",
        lambda: calls.append("operation"),
        warmup=2,
        iterations=4,
        synchronize=lambda: calls.append("synchronize"),
        clock=clock,
    )
    assert calls == [
        "operation",
        "operation",
        "synchronize",
        "clock",
        "operation",
        "operation",
        "operation",
        "operation",
        "synchronize",
        "clock",
    ]
    assert result == BenchmarkResult("step", 4, 4.0, 1.0)


@pytest.mark.parametrize(
    ("name", "warmup", "iterations", "message"),
    [
        ("", 0, 1, "name"),
        ("step", -1, 1, "warmup"),
        ("step", True, 1, "warmup"),
        ("step", 0, 0, "iterations"),
        ("step", 0, True, "iterations"),
    ],
)
def test_benchmark_crashes_on_invalid_configuration(
    name: str,
    warmup: int,
    iterations: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark(
            name,
            lambda: None,
            warmup=warmup,
            iterations=iterations,
            synchronize=lambda: None,
        )


def test_benchmark_crashes_on_non_monotonic_clock() -> None:
    times = iter([2.0, 1.0])
    with pytest.raises(ValueError, match="monotonic"):
        benchmark(
            "step",
            lambda: None,
            warmup=0,
            iterations=1,
            synchronize=lambda: None,
            clock=lambda: next(times),
        )


def test_write_result_serializes_exact_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    result = BenchmarkResult("step", 4, 2.0, 0.5)
    write_result(path, result)
    assert json.loads(path.read_text()) == {
        "iterations": 4,
        "mean_seconds": 0.5,
        "name": "step",
        "total_seconds": 2.0,
    }
    with pytest.raises(FileExistsError):
        write_result(path, result)
