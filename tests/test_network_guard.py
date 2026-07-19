"""The default suite must be provably offline, not offline by convention."""

from __future__ import annotations

import os
import socket

import pytest

OFFLINE_VARIABLES = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")


def test_default_suite_blocks_socket_connect() -> None:
    with pytest.raises(RuntimeError, match="network access is forbidden"):
        socket.create_connection(("127.0.0.1", 9))
    with socket.socket() as raw_socket, pytest.raises(
        RuntimeError, match="network access is forbidden"
    ):
        raw_socket.connect(("127.0.0.1", 9))


def test_default_suite_sets_offline_env_before_collection() -> None:
    for variable in OFFLINE_VARIABLES:
        assert os.environ[variable] == "1"
