from __future__ import annotations

from unittest.mock import Mock

import pytest
from datasets import Dataset, DatasetDict, Features

from concentration.data.safety_eval import (
    EXPECTED_SAFETY_FEATURES,
    EXPECTED_SAFETY_ROWS,
    assert_safety_schema,
    extract_safety_prompts,
    load_safety_eval_prompts,
)


def _fixture(
    prompts: list[str] | None = None,
    categories: list[str] | None = None,
    category_ids: list[int] | None = None,
) -> Dataset:
    if prompts is None and categories is None and category_ids is None:
        prompts = ["prompt a", "prompt b"]
        categories = ["animal_abuse", "privacy_violation"]
        category_ids = [0, 13]
    size = len(prompts or categories or category_ids or [])
    return Dataset.from_dict(
        {
            "prompt": prompts or [f"prompt {index}" for index in range(size)],
            "category": categories or ["animal_abuse"] * size,
            "category_id": category_ids or [0] * size,
        },
        features=EXPECTED_SAFETY_FEATURES,
    ).with_format("torch")


def test_safety_fixture_mirrors_real_schema_and_torch_format() -> None:
    fixture = _fixture()
    assert_safety_schema(fixture)
    assert fixture.format["type"] == "torch"
    prompts = extract_safety_prompts(fixture)
    assert [prompt.prompt for prompt in prompts] == ["prompt a", "prompt b"]
    assert [prompt.category_id for prompt in prompts] == [0, 13]


def test_safety_loader_asserts_container_splits_and_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sized = Dataset.from_dict(
        {
            "prompt": [f"prompt-{index}" for index in range(EXPECTED_SAFETY_ROWS)],
            "category": ["animal_abuse"] * EXPECTED_SAFETY_ROWS,
            "category_id": [0] * EXPECTED_SAFETY_ROWS,
        },
        features=EXPECTED_SAFETY_FEATURES,
    )
    loader = Mock(return_value={"not": "a dataset dict"})
    monkeypatch.setattr("concentration.data.safety_eval.load_dataset", loader)
    with pytest.raises(TypeError, match="DatasetDict"):
        load_safety_eval_prompts("fixture/id")
    loader.return_value = DatasetDict({"train": real_sized})
    with pytest.raises(ValueError, match="test split"):
        load_safety_eval_prompts("fixture/id")
    loader.return_value = DatasetDict({"test": _fixture()})
    with pytest.raises(ValueError, match="row count"):
        load_safety_eval_prompts("fixture/id")
    loader.return_value = DatasetDict({"test": real_sized})
    assert len(load_safety_eval_prompts("fixture/id")) == EXPECTED_SAFETY_ROWS


def test_safety_schema_assert_crashes_on_feature_and_order_drift() -> None:
    wrong = Dataset.from_dict(
        {"prompt": ["p"], "category": ["c"], "category_id": [0], "new": ["x"]}
    )
    with pytest.raises(ValueError, match="schema changed"):
        assert_safety_schema(wrong)
    reversed_names = list(reversed(EXPECTED_SAFETY_FEATURES))
    reordered = Dataset.from_dict(
        {name: [] for name in reversed_names},
        features=Features({name: EXPECTED_SAFETY_FEATURES[name] for name in reversed_names}),
    )
    with pytest.raises(ValueError, match="column order"):
        assert_safety_schema(reordered)


@pytest.mark.parametrize(
    "fixture",
    [
        Dataset.from_dict(
            {"prompt": [], "category": [], "category_id": []},
            features=EXPECTED_SAFETY_FEATURES,
        ),
        _fixture(prompts=[" "]),
        _fixture(categories=[" "]),
        _fixture(category_ids=[14]),
    ],
)
def test_extract_safety_prompts_crashes_on_invalid_content(fixture: Dataset) -> None:
    with pytest.raises(ValueError):
        extract_safety_prompts(fixture)


@pytest.mark.slow
def test_real_safety_eval_dataset_matches_recorded_schema_and_size() -> None:
    prompts = load_safety_eval_prompts()
    assert len(prompts) == EXPECTED_SAFETY_ROWS
    assert {prompt.category_id for prompt in prompts} == set(range(14))
