from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch as t
from hypothesis import given
from hypothesis import strategies as st
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3Config, Qwen3ForSequenceClassification

from concentration.config import DEV_REWARD_MODEL_ID, RewardModelConfig, WandbConfig
from concentration.models.reward_model import (
    CACHE_VERSION,
    INDEX_FILENAME,
    SCORES_FILENAME,
    ScoreCache,
    SequenceClassificationRewardModel,
    chat_template_hash,
    compute_score_diagnostics,
    log_score_diagnostics,
    render_reward_chat,
    score_cache_key,
)
from concentration.run_logging import RunLogger
from concentration.types import ScoreBatch


class FakeRewardTokenizer:
    chat_template = "fixture-template-v1"

    def __init__(self) -> None:
        self.render_calls: list[tuple[str, str, bool]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        self.render_calls.append((messages[0]["content"], messages[1]["content"], enable_thinking))
        return f"<user>{messages[0]['content']}</user><assistant>{messages[1]['content']}</assistant>"

    def __call__(
        self,
        rendered: list[str],
        *,
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> dict[str, t.Tensor]:
        assert padding is True
        assert truncation is False
        assert add_special_tokens is False
        assert return_tensors == "pt"
        lengths = [len(text) for text in rendered]
        width = max(lengths)
        input_ids = t.zeros((len(rendered), width), dtype=t.int64)
        attention_mask = t.zeros_like(input_ids)
        for row, text in enumerate(rendered):
            values = t.tensor([(ord(character) % 30) + 1 for character in text], dtype=t.int64)
            input_ids[row, : values.shape[0]] = values
            attention_mask[row, : values.shape[0]] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def tiny_reward_model() -> Qwen3ForSequenceClassification:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=256,
        pad_token_id=0,
    )
    config.num_labels = 1
    return Qwen3ForSequenceClassification(config).eval()


def test_sequence_classifier_returns_exact_raw_logits_in_batches() -> None:
    model = tiny_reward_model()
    tokenizer = FakeRewardTokenizer()
    reward_model = SequenceClassificationRewardModel(
        model,
        tokenizer,
        rm_id="fixture/rm",
        batch_size=2,
        max_length=200,
        device="cpu",
    )
    prompts = ["p0", "p1", "p2"]
    responses = ["r0", "r1", "r2"]
    scores = reward_model.score(prompts, responses)

    rendered = [
        f"<user>{prompt}</user><assistant>{response}</assistant>"
        for prompt, response in zip(prompts, responses, strict=True)
    ]
    first_batch = tokenizer(rendered[:2], padding=True, truncation=False, add_special_tokens=False, return_tensors="pt")
    second_batch = tokenizer(rendered[2:], padding=True, truncation=False, add_special_tokens=False, return_tensors="pt")
    with t.inference_mode():
        expected = t.cat([model(**first_batch).logits[:, 0], model(**second_batch).logits[:, 0]]).float()
    assert t.equal(scores, expected)
    assert scores.dtype == t.float32
    assert tokenizer.render_calls == [("p0", "r0", False), ("p1", "r1", False), ("p2", "r2", False)]
    assert reward_model.rm_id == "fixture/rm"
    assert reward_model.template_hash == hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    assert model.training is False
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_reward_scoring_crashes_on_invalid_batches_and_length() -> None:
    reward_model = SequenceClassificationRewardModel(
        tiny_reward_model(),
        FakeRewardTokenizer(),
        rm_id="fixture/rm",
        batch_size=2,
        max_length=20,
        device="cpu",
    )
    with pytest.raises(ValueError, match="equal lengths"):
        reward_model.score(["p"], [])
    with pytest.raises(ValueError, match="non-empty"):
        reward_model.score([], [])
    with pytest.raises(ValueError, match="exceeds"):
        reward_model.score(["prompt"], ["response"])
    with pytest.raises(ValueError, match="non-blank"):
        render_reward_chat(FakeRewardTokenizer(), " ", "response")


class WrongShapeModel(nn.Module):
    def forward(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(logits=t.ones((1, 2)))


def test_reward_scoring_crashes_on_non_scalar_classifier() -> None:
    reward_model = SequenceClassificationRewardModel(
        WrongShapeModel(),
        FakeRewardTokenizer(),
        rm_id="fixture/rm",
        batch_size=1,
        max_length=200,
        device="cpu",
    )
    with pytest.raises(ValueError, match=r"\[batch, 1\]"):
        reward_model.score(["p"], ["r"])


def test_reward_model_constructor_and_template_hash_validation() -> None:
    tokenizer = FakeRewardTokenizer()
    with pytest.raises(ValueError, match="rm_id"):
        SequenceClassificationRewardModel(tiny_reward_model(), tokenizer, rm_id="", batch_size=1, max_length=10, device="cpu")
    with pytest.raises(ValueError, match="batch_size"):
        SequenceClassificationRewardModel(tiny_reward_model(), tokenizer, rm_id="rm", batch_size=0, max_length=10, device="cpu")
    with pytest.raises(ValueError, match="max_length"):
        SequenceClassificationRewardModel(tiny_reward_model(), tokenizer, rm_id="rm", batch_size=1, max_length=0, device="cpu")
    with pytest.raises(ValueError, match="device"):
        SequenceClassificationRewardModel(tiny_reward_model(), tokenizer, rm_id="rm", batch_size=1, max_length=10, device="")
    tokenizer.chat_template = ""
    with pytest.raises(ValueError, match="chat template"):
        chat_template_hash(tokenizer)


def test_reward_chat_and_tokenizer_output_contracts_crash_loudly() -> None:
    tokenizer = FakeRewardTokenizer()
    tokenizer.apply_chat_template = Mock(return_value=123)
    with pytest.raises(TypeError, match="render"):
        render_reward_chat(tokenizer, "prompt", "response")

    class InvalidTokenizer(FakeRewardTokenizer):
        def __call__(
            self,
            rendered: list[str],
            *,
            padding: bool,
            truncation: bool,
            add_special_tokens: bool,
            return_tensors: str,
        ) -> dict[str, t.Tensor]:
            del rendered, padding, truncation, add_special_tokens, return_tensors
            return {"input_ids": cast(Any, [1, 2])}

    invalid_tokenizer = InvalidTokenizer()
    reward_model = SequenceClassificationRewardModel(
        tiny_reward_model(),
        invalid_tokenizer,
        rm_id="fixture/rm",
        batch_size=1,
        max_length=20,
        device="cpu",
    )
    with pytest.raises(TypeError, match="rank-2"):
        reward_model.score(["p"], ["r"])


class WrongBatchModel(nn.Module):
    def forward(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(logits=t.ones((1, 1)))


def test_reward_scoring_crashes_when_model_returns_wrong_batch_size() -> None:
    reward_model = SequenceClassificationRewardModel(
        WrongBatchModel(),
        FakeRewardTokenizer(),
        rm_id="fixture/rm",
        batch_size=2,
        max_length=200,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="wrong number"):
        reward_model.score(["p0", "p1"], ["r0", "r1"])


def test_reward_model_from_config_uses_exact_loader_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = FakeRewardTokenizer()
    model = Mock(spec=nn.Module)
    model.to.return_value = model
    model.eval.return_value = model
    model.parameters.return_value = []
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(
        "concentration.models.reward_model.AutoTokenizer.from_pretrained",
        tokenizer_loader,
    )
    monkeypatch.setattr(
        "concentration.models.reward_model.AutoModelForSequenceClassification.from_pretrained",
        model_loader,
    )
    config = RewardModelConfig.from_raw(
        model_id="org/rm",
        revision="commit",
        dtype="float32",
        device="cpu",
        batch_size=3,
        max_length=64,
    )
    loaded = SequenceClassificationRewardModel.from_config(config)
    tokenizer_loader.assert_called_once_with("org/rm", revision="commit")
    model_loader.assert_called_once_with("org/rm", revision="commit", dtype=t.float32)
    assert loaded.rm_id == "org/rm"


@pytest.mark.property
@given(
    rm_id=st.text(min_size=1, max_size=20),
    template_hash=st.text(min_size=1, max_size=20),
    prompt=st.text(max_size=20),
    response=st.text(max_size=20),
)
def test_score_cache_key_is_exact_stable_sha256(
    rm_id: str,
    template_hash: str,
    prompt: str,
    response: str,
) -> None:
    expected = hashlib.sha256((rm_id + template_hash + prompt + response).encode()).hexdigest()
    assert score_cache_key(rm_id, template_hash, prompt, response) == expected


def _build_cache(path: Path) -> ScoreCache:
    return ScoreCache.build(
        path,
        rm_id="fixture/rm",
        template_hash="a" * 64,
        prompts=["p0", "p1", "p2"],
        responses=["r0", "r1", "r2"],
        scores=ScoreBatch.from_tensor(t.tensor([-2.5, 0.0, 7.25], dtype=t.float32)),
    )


def test_score_cache_round_trip_lookup_and_json_index(tmp_path: Path) -> None:
    cache = _build_cache(tmp_path / "cache")
    loaded = ScoreCache.load(tmp_path / "cache")
    assert t.equal(loaded.scores.tensor, t.tensor([-2.5, 0.0, 7.25]))
    assert t.equal(
        cache.lookup(["p2", "p0"], ["r2", "r0"]).tensor,
        t.tensor([7.25, -2.5]),
    )
    payload = json.loads((tmp_path / "cache" / INDEX_FILENAME).read_text())
    assert payload["version"] == CACHE_VERSION
    assert payload["rm_id"] == "fixture/rm"
    assert set(payload["index"].values()) == {0, 1, 2}


def test_score_cache_missing_key_completeness_and_metadata_crash(tmp_path: Path) -> None:
    cache = _build_cache(tmp_path / "cache")
    with pytest.raises(KeyError, match="missing reward score"):
        cache.lookup(["missing"], ["response"])
    with pytest.raises(KeyError, match="incomplete"):
        cache.verify("fixture/rm", "a" * 64, ["p0", "p1"], ["r0", "r1"])
    with pytest.raises(ValueError, match="rm_id mismatch"):
        cache.verify("other/rm", "a" * 64, ["p0", "p1", "p2"], ["r0", "r1", "r2"])
    with pytest.raises(ValueError, match="template hash mismatch"):
        cache.verify("fixture/rm", "b" * 64, ["p0", "p1", "p2"], ["r0", "r1", "r2"])
    with pytest.raises(ValueError, match="equal lengths"):
        cache.verify("fixture/rm", "a" * 64, ["p0"], [])
    with pytest.raises(ValueError, match="unique"):
        cache.verify("fixture/rm", "a" * 64, ["p0", "p0"], ["r0", "r0"])
    with pytest.raises(ValueError, match="equal lengths"):
        cache.lookup(["p0"], [])
    with pytest.raises(ValueError, match="non-empty"):
        cache.lookup([], [])


def test_score_cache_build_rejects_mismatches_duplicates_and_existing_path(tmp_path: Path) -> None:
    scores = ScoreBatch.from_tensor(t.tensor([1.0, 2.0], dtype=t.float32))
    with pytest.raises(ValueError, match="equal lengths"):
        ScoreCache.build(
            tmp_path / "bad",
            rm_id="rm",
            template_hash="a" * 64,
            prompts=["p"],
            responses=["r"],
            scores=scores,
        )
    with pytest.raises(ValueError, match="unique"):
        ScoreCache.build(
            tmp_path / "duplicates",
            rm_id="rm",
            template_hash="a" * 64,
            prompts=["p", "p"],
            responses=["r", "r"],
            scores=scores,
        )
    _build_cache(tmp_path / "cache")
    with pytest.raises(FileExistsError):
        _build_cache(tmp_path / "cache")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"version": 999}, "version"),
        ({"rm_id": ""}, "rm_id"),
        ({"chat_template_hash": "short"}, "hash"),
        ({"index": {"short": 0}}, "index"),
        ({"index": {"a" * 64: 3, "b" * 64: 4, "c" * 64: 5}}, "cover"),
        ({"extra": 1}, "unexpected"),
    ],
)
def test_score_cache_load_crashes_on_corrupt_json(
    mutation: dict[str, object],
    message: str,
    tmp_path: Path,
) -> None:
    _build_cache(tmp_path / "cache")
    path = tmp_path / "cache" / INDEX_FILENAME
    payload = json.loads(path.read_text())
    if "extra" in mutation:
        payload.update(mutation)
    else:
        payload.update(mutation)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        ScoreCache.load(tmp_path / "cache")


def test_score_cache_load_crashes_on_wrong_tensor_set(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    save_file({"wrong": t.ones(3)}, tmp_path / "cache" / SCORES_FILENAME)
    with pytest.raises(ValueError, match="exactly"):
        ScoreCache.load(tmp_path / "cache")


def test_raw_score_diagnostics_do_not_modify_or_replace_scores() -> None:
    tensor = t.tensor([-2.0, 1.0, 4.0], dtype=t.float32)
    scores = ScoreBatch.from_tensor(tensor)
    before = tensor.clone()
    diagnostics = compute_score_diagnostics(scores)
    assert diagnostics.mean == pytest.approx(1.0)
    assert diagnostics.standard_deviation == pytest.approx(6.0**0.5)
    assert diagnostics.count == 3
    assert scores.tensor is tensor
    assert t.equal(scores.tensor, before)


def test_raw_score_diagnostics_are_logged_without_transforming_targets(tmp_path: Path) -> None:
    scores = ScoreBatch.from_tensor(t.tensor([-2.0, 1.0, 4.0], dtype=t.float32))
    before = scores.tensor.clone()
    path = tmp_path / "run.jsonl"
    with RunLogger(
        path,
        WandbConfig.from_raw(mode="disabled"),
        run_name="diagnostics",
        run_config={},
    ) as logger:
        returned = log_score_diagnostics(logger, scores, step=3, split="train")
    assert returned is scores
    assert t.equal(returned.tensor, before)
    assert json.loads(path.read_text()) == {
        "step": 3,
        "train/rm_score_count": 3,
        "train/rm_score_mean": 1.0,
        "train/rm_score_std": pytest.approx(6.0**0.5),
    }
    with pytest.raises(ValueError, match="split"):
        log_score_diagnostics(Mock(spec=RunLogger), scores, step=0, split=" ")


@pytest.mark.slow
def test_real_skywork_dev_reward_model_loads_and_returns_raw_score() -> None:
    config = RewardModelConfig.from_raw(
        model_id=DEV_REWARD_MODEL_ID,
        dtype="float32",
        device="cpu",
        batch_size=1,
        max_length=512,
    )
    reward_model = SequenceClassificationRewardModel.from_config(config)
    scores = reward_model.score(
        ["Give one harmless greeting."],
        ["Hello! I hope you have a wonderful day."],
    )
    assert scores.shape == (1,)
    assert scores.dtype == t.float32
    assert bool(t.isfinite(scores).all())
    assert reward_model.rm_id == DEV_REWARD_MODEL_ID
    del scores, reward_model
    gc.collect()
