"""JSONL-always experiment logging with optional Weights & Biases mirroring."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

import wandb
from beartype import beartype

from concentration.config import WandbConfig, WandbMode

MetricValue = bool | int | float | str | None


class RunLogger:
    """Write every metric record to JSONL and optionally mirror it to wandb."""

    @beartype
    def __init__(
        self,
        path: Path,
        wandb_config: WandbConfig,
        *,
        run_name: str,
        run_config: Mapping[str, MetricValue],
    ) -> None:
        if not run_name.strip():
            raise ValueError("run_name must be non-empty")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("x", encoding="utf-8")
        self._closed = False
        self._wandb_run: Any | None = None
        if wandb_config.mode is not WandbMode.DISABLED:
            self._wandb_run = wandb.init(
                project=wandb_config.project,
                name=run_name,
                mode=wandb_config.mode.value,
                config=dict(run_config),
            )

    @beartype
    def log(self, step: int, metrics: Mapping[str, MetricValue]) -> None:
        """Append one finite, JSON-serializable metric record."""
        if self._closed:
            raise RuntimeError("cannot log after logger is closed")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        if "step" in metrics:
            raise ValueError("metrics must not contain the reserved key 'step'")
        for name, value in metrics.items():
            if not name:
                raise ValueError("metric names must be non-empty")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metric {name!r} must be finite")
        record: dict[str, MetricValue] = {"step": step, **metrics}
        self._stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._stream.flush()
        if self._wandb_run is not None:
            self._wandb_run.log(dict(metrics), step=step)

    def close(self) -> None:
        """Flush and close both logging sinks."""
        if self._closed:
            return
        self._stream.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
        self._closed = True

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
