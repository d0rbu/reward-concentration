from __future__ import annotations

import os
import random
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from concentration.seeding import seed_all
from concentration.types import parse_seed


def test_seed_all_repeats_python_and_torch_streams() -> None:
    seed = parse_seed(123)
    seed_all(seed)
    first_python = [random.random() for _ in range(4)]
    first_torch = t.rand(4)
    seed_all(seed)
    assert [random.random() for _ in range(4)] == first_python
    assert t.equal(t.rand(4), first_torch)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_seed_all_calls_cuda_and_transformers_seeders(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        "concentration.seeding.t.cuda.manual_seed_all",
        lambda value: calls.append(("cuda", value)),
    )
    monkeypatch.setattr(
        "concentration.seeding.set_seed",
        lambda value: calls.append(("transformers", value)),
    )
    seed_all(parse_seed(7))
    assert calls.count(("cuda", 7)) >= 1
    assert calls[-1] == ("transformers", 7)


def test_seed_all_requires_a_refined_valid_seed() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        seed_all(cast(Any, -1))
