from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from concentration.config import WandbConfig
from concentration.run_logging import MetricValue, RunLogger


def test_disabled_logger_always_writes_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init = Mock()
    monkeypatch.setattr("concentration.run_logging.wandb.init", init)
    path = tmp_path / "nested" / "run.jsonl"
    with RunLogger(
        path,
        WandbConfig.from_raw(mode="disabled"),
        run_name="unit",
        run_config={"seed": 0},
    ) as logger:
        logger.log(0, {"loss": 1.5, "ok": True})
        logger.log(1, {"label": "done", "optional": None})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {"loss": 1.5, "ok": True, "step": 0},
        {"label": "done", "optional": None, "step": 1},
    ]
    init.assert_not_called()


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_enabled_logger_mirrors_metrics_to_wandb(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock()
    init = Mock(return_value=run)
    monkeypatch.setattr("concentration.run_logging.wandb.init", init)
    logger = RunLogger(
        tmp_path / f"{mode}.jsonl",
        WandbConfig.from_raw(mode=mode, project="project"),
        run_name="trial",
        run_config={"seed": 3},
    )
    logger.log(2, {"metric": 4.0})
    logger.close()
    logger.close()
    init.assert_called_once_with(
        project="project",
        name="trial",
        mode=mode,
        config={"seed": 3},
    )
    run.log.assert_called_once_with({"metric": 4.0}, step=2)
    run.finish.assert_called_once_with()


@pytest.mark.parametrize(
    ("step", "metrics", "message"),
    [
        (-1, {"loss": 1.0}, "step"),
        (True, {"loss": 1.0}, "step"),
        (0, {"step": 1}, "reserved"),
        (0, {"": 1}, "names"),
        (0, {"loss": float("nan")}, "finite"),
    ],
)
def test_logger_crashes_on_invalid_records(
    step: int,
    metrics: dict[str, MetricValue],
    message: str,
    tmp_path: Path,
) -> None:
    logger = RunLogger(
        tmp_path / f"{message}.jsonl",
        WandbConfig.from_raw(mode="disabled"),
        run_name="unit",
        run_config={},
    )
    with pytest.raises((TypeError, ValueError), match=message):
        logger.log(step, metrics)
    logger.close()


def test_logger_crashes_on_reused_path_empty_name_and_log_after_close(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    logger = RunLogger(
        path,
        WandbConfig.from_raw(mode="disabled"),
        run_name="unit",
        run_config={},
    )
    logger.close()
    with pytest.raises(FileExistsError):
        RunLogger(
            path,
            WandbConfig.from_raw(mode="disabled"),
            run_name="unit",
            run_config={},
        )
    with pytest.raises(ValueError, match="run_name"):
        RunLogger(
            tmp_path / "other.jsonl",
            WandbConfig.from_raw(mode="disabled"),
            run_name=" ",
            run_config={},
        )
    with pytest.raises(RuntimeError, match="closed"):
        logger.log(0, {"loss": 1.0})
