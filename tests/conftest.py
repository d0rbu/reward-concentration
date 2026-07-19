from __future__ import annotations

import random
from typing import Any, cast

import pytest
import torch as t


def pytest_configure(config: pytest.Config) -> None:
    """The explicit slow-only suite validates integrations, not the fast-suite coverage gate."""
    if config.option.markexpr.strip() == "slow":  # pragma: no cover - slow-only invocation
        coverage_plugin = cast(Any, config.pluginmanager.get_plugin("_cov"))
        config.option.no_cov = True
        coverage_plugin.options.no_cov = True
        coverage_plugin._disabled = True


@pytest.fixture(autouse=True)
def deterministic_isolated_test_environment(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Make fast tests deterministic, wandb-disabled, and Hugging Face offline."""
    random.seed(0)
    t.manual_seed(0)
    t.use_deterministic_algorithms(True)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    offline_variables = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    if request.node.get_closest_marker("slow") is None:
        for variable in offline_variables:
            monkeypatch.setenv(variable, "1")
    else:  # pragma: no cover - exercised only by the separately run slow suite
        for variable in offline_variables:
            monkeypatch.delenv(variable, raising=False)
