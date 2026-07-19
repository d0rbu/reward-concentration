from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from datasets import Dataset, DatasetDict, Features, Value
from hypothesis import given
from hypothesis import strategies as st
from transformers import AutoTokenizer

from concentration.config import DEV_POLICY_MODEL_ID, DataConfig
from concentration.data.preference import (
    EXPECTED_PREFERENCE_FEATURES,
    OBSERVED_PREFERENCE_ROWS,
    PreferencePair,
    PreferenceResponse,
    PreferenceSplit,
    PreferenceSplits,
    TokenSpan,
    assert_preference_schema,
    build_prompt_splits,
    load_preference_dataset,
    tokenize_chat,
    tokenize_preference_splits,
)


class CharacterChatTokenizer:
    """Offset-aware tokenizer whose token IDs map one-to-one to rendered characters."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        continue_final_message: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        assert enable_thinking is False
        pieces: list[str] = []
        for index, message in enumerate(messages):
            pieces.extend((f"<{message['role']}>", message["content"]))
            is_continued = continue_final_message and index == len(messages) - 1
            if not is_continued:
                pieces.append("</turn>")
        return "".join(pieces)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list[Any]]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) + 1 for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return "".join(chr(token_id - 1) for token_id in token_ids)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source_digest(prompt: str, response: str) -> str:
    return _digest(prompt + response)


def _row(prompt: str, response_0: str, response_1: str, better: int = 0) -> dict[str, object]:
    return {
        "prompt": prompt,
        "response_0": response_0,
        "response_1": response_1,
        "prompt_source": "fixture-prompt",
        "response_0_source": "fixture-response",
        "response_1_source": "fixture-response",
        "better_response_id": better,
        "response_0_sha256": _source_digest(prompt, response_0),
        "response_1_sha256": _source_digest(prompt, response_1),
    }


def _dataset(rows: list[dict[str, object]]) -> Dataset:
    columns = {name: [row[name] for row in rows] for name in EXPECTED_PREFERENCE_FEATURES}
    return Dataset.from_dict(columns, features=EXPECTED_PREFERENCE_FEATURES).with_format("torch")


def preference_fixture(*, include_blank: bool = False) -> DatasetDict:
    train_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for index in range(12):
        prompt = f"prompt-{index}"
        target = train_rows if index < 8 else test_rows
        target.append(_row(prompt, f"good-{index}", f"bad-{index}", index % 2))
    # Preserve an overlapping upstream prompt and a response whose pair-relative label flips.
    test_rows.append(_row("prompt-0", "good-0", "alternative-0", 1))
    if include_blank:
        train_rows.append(_row("prompt-1", "", "nonblank-extra", 1))
    return DatasetDict({"train": _dataset(train_rows), "test": _dataset(test_rows)})


def test_fixture_mirrors_real_schema_and_torch_format() -> None:
    fixture = preference_fixture()
    assert_preference_schema(fixture)
    assert fixture["train"].features == EXPECTED_PREFERENCE_FEATURES
    assert fixture["train"].format["type"] == "torch"
    assert fixture["train"][0]["better_response_id"].ndim == 0


def test_preference_loader_materializes_and_asserts_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = preference_fixture()
    loader = Mock(return_value=fixture)
    monkeypatch.setattr("concentration.data.preference.load_dataset", loader)
    assert load_preference_dataset("fixture/id") is fixture
    loader.assert_called_once_with("fixture/id")
    loader.return_value = fixture["train"]
    with pytest.raises(TypeError, match="DatasetDict"):
        load_preference_dataset("fixture/id")


def test_schema_assert_crashes_on_split_feature_and_order_drift() -> None:
    fixture = preference_fixture()
    with pytest.raises(ValueError, match="splits changed"):
        assert_preference_schema(DatasetDict({"train": fixture["train"]}))

    wrong_features = Features({**dict(EXPECTED_PREFERENCE_FEATURES), "new": Value("string")})
    wrong = Dataset.from_dict(
        {**{name: [] for name in EXPECTED_PREFERENCE_FEATURES}, "new": []},
        features=wrong_features,
    )
    with pytest.raises(ValueError, match="schema changed"):
        assert_preference_schema(DatasetDict({"train": wrong, "test": wrong}))

    reversed_names = list(reversed(EXPECTED_PREFERENCE_FEATURES))
    reordered = Dataset.from_dict(
        {name: [] for name in reversed_names},
        features=Features({name: EXPECTED_PREFERENCE_FEATURES[name] for name in reversed_names}),
    )
    with pytest.raises(ValueError, match="column order"):
        assert_preference_schema(DatasetDict({"train": reordered, "test": reordered}))


def test_prompt_splits_are_seeded_disjoint_and_merge_upstream_splits() -> None:
    fixture = preference_fixture()
    config = DataConfig.from_raw(
        heldout_probe_train_frac=0.25,
        heldout_probe_test_frac=0.25,
        seed=17,
    )
    first = build_prompt_splits(fixture, config)
    second = build_prompt_splits(fixture, config)
    assert first == second
    prompt_sets = [{response.prompt for response in split.responses} for _, split in first.items()]
    assert prompt_sets[0].isdisjoint(prompt_sets[1])
    assert prompt_sets[0].isdisjoint(prompt_sets[2])
    assert prompt_sets[1].isdisjoint(prompt_sets[2])
    assert set.union(*prompt_sets) == {f"prompt-{index}" for index in range(12)}
    split_with_prompt_zero = next(
        split for _, split in first.items() if any(pair.prompt == "prompt-0" for pair in split.pairs)
    )
    prompt_zero_pairs = [pair for pair in split_with_prompt_zero.pairs if pair.prompt == "prompt-0"]
    assert len(prompt_zero_pairs) == 2
    assert {pair.better_response_id for pair in prompt_zero_pairs} == {0, 1}
    assert sum(response.response == "good-0" for response in split_with_prompt_zero.responses) == 1
    assert split_with_prompt_zero.duplicate_responses_removed == 1


def test_blank_responses_are_counted_and_pairs_are_removed(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    splits = build_prompt_splits(
        preference_fixture(include_blank=True),
        DataConfig.from_raw(
            heldout_probe_train_frac=0.25,
            heldout_probe_test_frac=0.25,
            seed=0,
        ),
    )
    assert sum(split.blank_responses_removed for _, split in splits.items()) == 1
    assert sum(split.pairs_with_blank_response for _, split in splits.items()) == 1
    assert all(response.response.strip() for _, split in splits.items() for response in split.responses)
    assert "blank_responses_removed" in caplog.text


def test_prompt_split_crashes_on_invalid_source_values_and_empty_partition() -> None:
    fixture = preference_fixture()
    bad_hash_rows = [_row("p", "a", "b")]
    bad_hash_rows[0]["response_0_sha256"] = "bad"
    invalid = DatasetDict(
        {"train": _dataset(bad_hash_rows), "test": _dataset([_row("q", "c", "d")])}
    )
    with pytest.raises(ValueError, match="sha256"):
        build_prompt_splits(
            invalid,
            DataConfig.from_raw(
                heldout_probe_train_frac=0.25,
                heldout_probe_test_frac=0.25,
            ),
        )

    bad_id_rows = [_row("p", "a", "b")]
    bad_id_rows[0]["better_response_id"] = 2
    invalid_id = DatasetDict(
        {"train": _dataset(bad_id_rows), "test": _dataset([_row("q", "c", "d")])}
    )
    with pytest.raises(ValueError, match="better_response_id"):
        build_prompt_splits(
            invalid_id,
            DataConfig.from_raw(
                heldout_probe_train_frac=0.25,
                heldout_probe_test_frac=0.25,
            ),
        )

    with pytest.raises(ValueError, match="empty split"):
        build_prompt_splits(
            fixture,
            DataConfig.from_raw(
                heldout_probe_train_frac=0.0,
                heldout_probe_test_frac=0.25,
            ),
        )


def test_prompt_split_crashes_on_blank_prompt_and_non_string_text() -> None:
    blank = DatasetDict(
        {
            "train": _dataset([_row(" ", "a", "b")]),
            "test": _dataset([_row("q", "c", "d")]),
        }
    )
    with pytest.raises(ValueError, match="prompts"):
        build_prompt_splits(
            blank,
            DataConfig.from_raw(
                heldout_probe_train_frac=0.25,
                heldout_probe_test_frac=0.25,
            ),
        )


def test_chat_spans_include_end_turn_only_in_loss_span() -> None:
    tokenizer = CharacterChatTokenizer()
    response = "assistant content"
    encoded = tokenize_chat(tokenizer, "user prompt", response)
    pool_ids = encoded.input_ids[encoded.pool_span.start : encoded.pool_span.stop].tolist()
    loss_ids = encoded.input_ids[encoded.loss_span.start : encoded.loss_span.stop].tolist()
    assert tokenizer.decode(pool_ids) == response
    assert tokenizer.decode(loss_ids) == response + "</turn>"
    assert encoded.attention_mask.dtype == encoded.attention_mask.bool().dtype
    assert encoded.attention_mask.all()


@pytest.mark.property
@given(
    prompt=st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=30),
    response=st.text(
        alphabet=st.characters(min_codepoint=33, max_codepoint=126),
        min_size=1,
        max_size=30,
    ),
)
def test_chat_pool_span_round_trips_random_content(prompt: str, response: str) -> None:
    tokenizer = CharacterChatTokenizer()
    encoded = tokenize_chat(tokenizer, prompt, response)
    ids = encoded.input_ids[encoded.pool_span.start : encoded.pool_span.stop].tolist()
    assert tokenizer.decode(ids) == response
    assert encoded.loss_span.start == encoded.pool_span.start
    assert encoded.loss_span.stop > encoded.pool_span.stop


@pytest.mark.parametrize(("start", "stop"), [(-1, 2), (1, 1), (2, 1)])
def test_token_span_rejects_invalid_boundaries(start: int, stop: int) -> None:
    with pytest.raises(ValueError):
        TokenSpan(start, stop)
    with pytest.raises(TypeError):
        TokenSpan(True, 2)  # type: ignore[arg-type]


@pytest.mark.parametrize(("prompt", "response"), [("", "response"), ("prompt", " ")])
def test_chat_tokenizer_crashes_on_blank_content(prompt: str, response: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        tokenize_chat(CharacterChatTokenizer(), prompt, response)


def test_tokenization_drops_overlong_items_logs_counts_and_keeps_splits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    splits = build_prompt_splits(
        preference_fixture(),
        DataConfig.from_raw(
            max_len=10_000,
            heldout_probe_train_frac=0.25,
            heldout_probe_test_frac=0.25,
            seed=0,
        ),
    )
    target_name, target_split = next(iter(splits.items()))
    target_response = target_split.responses[0]
    long_response = replace(
        target_response,
        response="x" * 500,
        response_sha256=_source_digest(target_response.prompt, "x" * 500),
    )
    old_key = target_response.key
    updated_pairs = tuple(
        replace(
            pair,
            response_0=long_response.response,
            response_0_sha256=long_response.response_sha256,
        )
        if (pair.prompt, pair.response_0) == old_key
        else pair
        for pair in target_split.pairs
    )
    updated_split = replace(
        target_split,
        pairs=updated_pairs,
        responses=(long_response, *target_split.responses[1:]),
    )
    modified = replace(splits, **{target_name: updated_split})
    tokenized = tokenize_preference_splits(
        modified,
        CharacterChatTokenizer(),
        DataConfig.from_raw(
            max_len=100,
            heldout_probe_train_frac=0.25,
            heldout_probe_test_frac=0.25,
            seed=0,
        ),
    )
    tokenized_target = dict(tokenized.items())[target_name]
    assert tokenized_target.overlong_responses_removed == 1
    assert tokenized_target.incomplete_pairs_removed >= 1
    assert "overlong_removed=1" in caplog.text


def test_tokenization_crashes_when_a_split_empties() -> None:
    response_0 = PreferenceResponse(
        "prompt",
        "a" * 100,
        _source_digest("prompt", "a" * 100),
    )
    response_1 = PreferenceResponse(
        "prompt",
        "b" * 100,
        _source_digest("prompt", "b" * 100),
    )
    pair = PreferencePair(
        "prompt",
        response_0.response,
        response_1.response,
        response_0.response_sha256,
        response_1.response_sha256,
        0,
    )
    split = PreferenceSplit((pair,), (response_0, response_1), 0, 0, 0)
    splits = PreferenceSplits(split, split, split)
    with pytest.raises(ValueError, match="empty after length"):
        tokenize_preference_splits(
            splits,
            CharacterChatTokenizer(),
            DataConfig.from_raw(
                max_len=20,
                heldout_probe_train_frac=0.25,
                heldout_probe_test_frac=0.25,
            ),
        )


@pytest.mark.slow
def test_real_preference_dataset_matches_recorded_schema_and_sizes() -> None:
    dataset = load_preference_dataset()
    assert {split: len(values) for split, values in dataset.items()} == OBSERVED_PREFERENCE_ROWS


@pytest.mark.slow
def test_real_preference_rows_validate_and_split_without_prompt_leakage() -> None:
    dataset = load_preference_dataset()
    splits = build_prompt_splits(dataset, DataConfig.from_raw(seed=0))
    prompt_sets = [{response.prompt for response in split.responses} for _, split in splits.items()]
    assert set.union(*prompt_sets) == set(dataset["train"]["prompt"]) | set(dataset["test"]["prompt"])
    assert sum(split.blank_responses_removed for _, split in splits.items()) == 40
    assert sum(split.duplicate_responses_removed for _, split in splits.items()) == 4_646
    assert prompt_sets[0].isdisjoint(prompt_sets[1])
    assert prompt_sets[0].isdisjoint(prompt_sets[2])
    assert prompt_sets[1].isdisjoint(prompt_sets[2])


@pytest.mark.slow
def test_real_qwen3_chat_template_produces_exact_response_spans() -> None:
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(DEV_POLICY_MODEL_ID))
    response = "A short test response."
    encoded = tokenize_chat(tokenizer, "Give a short response.", response)
    pool_ids = encoded.input_ids[encoded.pool_span.start : encoded.pool_span.stop]
    assert tokenizer.decode(pool_ids, skip_special_tokens=False) == response
    loss_ids = encoded.input_ids[encoded.loss_span.start : encoded.loss_span.stop]
    assert loss_ids.shape[0] > pool_ids.shape[0]
