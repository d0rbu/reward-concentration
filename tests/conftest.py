from __future__ import annotations

import os
import random
import socket
from typing import Any, cast

import pytest
import torch as t
from hypothesis import settings

settings.register_profile("ci", derandomize=True, print_blob=True)
if os.environ.get("CI") == "true":  # pragma: no cover - exercised only on CI runners
    settings.load_profile("ci")

OFFLINE_VARIABLES = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")


def pytest_configure(config: pytest.Config) -> None:
    """Pin the network posture before test modules import Hugging Face libraries.

    huggingface_hub and datasets snapshot the offline env vars at import time, so a
    per-test fixture is too late; this hook runs before collection imports anything.
    The explicit slow-only suite validates live integrations, not the fast-suite
    coverage gate, so it runs online and without coverage.
    """
    if config.option.markexpr.strip() == "slow":  # pragma: no cover - slow-only invocation
        for variable in OFFLINE_VARIABLES:
            os.environ.pop(variable, None)
        coverage_plugin = cast(Any, config.pluginmanager.get_plugin("_cov"))
        config.option.no_cov = True
        coverage_plugin.options.no_cov = True
        coverage_plugin._disabled = True
    else:
        for variable in OFFLINE_VARIABLES:
            os.environ[variable] = "1"


@pytest.fixture(autouse=True)
def deterministic_isolated_test_environment(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Make fast tests deterministic, wandb-disabled, and socket-free."""
    random.seed(0)
    t.manual_seed(0)
    t.use_deterministic_algorithms(True)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if request.node.get_closest_marker("slow") is None:

        def blocked_connect(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("network access is forbidden in the default test suite")

        monkeypatch.setattr(socket.socket, "connect", blocked_connect)
        monkeypatch.setattr(socket, "create_connection", blocked_connect)
