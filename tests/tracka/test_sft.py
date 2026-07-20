from __future__ import annotations

import gc
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from hypothesis import given
from hypothesis import strategies as st
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

import concentration.tracka.sft as sft_module
from concentration.config import (
    DEV_POLICY_MODEL_ID,
    DataConfig,
    ModelConfig,
    SFTTrainConfig,
    WandbConfig,
)
from concentration.data.preference import (
    PreferencePair,
    PreferenceResponse,
    PreferenceSplit,
    PreferenceSplits,
    TokenizedConversation,
    TokenizedPreferenceResponse,
    TokenizedPreferenceSplit,
    TokenizedPreferenceSplits,
    TokenSpan,
    build_prompt_splits,
    load_preference_dataset,
    tokenize_preference_splits,
)
from concentration.eval.capability import heldout_response_perplexity
from concentration.models.policy import load_policy
from concentration.tracka.sft import (
    EFFECTIVE_CONFIG_FILENAME,
    IGNORE_INDEX,
    RUN_MANIFEST_FILENAME,
    TRAIN_LOGS_FILENAME,
    SFTDataCollator,
    SFTDatasetCounts,
    SFTResult,
    build_sft_dataset,
    effective_config,
    labels_for_conversation,
    mask_loss_spans,
    train_sft,
    warmup_steps_for,
)
from concentration.types import parse_unit_interval


def _digest(prompt: str, response: str) -> str:
    return hashlib.sha256((prompt + response).encode()).hexdigest()


def _pair(prompt: str, response_0: str, response_1: str, better: int) -> PreferencePair:
    return PreferencePair(
        prompt=prompt,
        response_0=response_0,
        response_1=response_1,
        response_0_sha256=_digest(prompt, response_0),
        response_1_sha256=_digest(prompt, response_1),
        better_response_id=better,
    )


def _tokenized_response(
    prompt: str,
    response: str,
    response_token: int,
) -> TokenizedPreferenceResponse:
    source = PreferenceResponse(prompt, response, _digest(prompt, response))
    conversation = TokenizedConversation(
        input_ids=t.tensor([3, 4, response_token, 2], dtype=t.int64),
        attention_mask=t.ones(4, dtype=t.bool),
        loss_span=TokenSpan(2, 4),
        pool_span=TokenSpan(2, 3),
    )
    return TokenizedPreferenceResponse(source, conversation)


def tokenized_preference_fixture() -> TokenizedPreferenceSplits:
    """Fixture with pairwise-distinct provenance counts and set-op-distinct labels.

    pairs=5, unique=6, overlong=7, incomplete=8, items=3, duplicates=2, dual=1; the
    dual-label intersection (1) differs from both set differences (2 and 2), so swapped
    count fields and swapped set operations both change observable values.
    """
    resp_a = _tokenized_response("train-a", "resp-a", 5)
    resp_b = _tokenized_response("train-a", "resp-b", 6)
    resp_c = _tokenized_response("train-a", "resp-c", 7)
    resp_d = _tokenized_response("train-a", "resp-d", 8)
    resp_e = _tokenized_response("train-a", "resp-e", 9)
    unpaired = _tokenized_response("train-a", "unpaired", 10)
    train = TokenizedPreferenceSplit(
        pairs=(
            _pair("train-a", "resp-a", "resp-d", 0),
            _pair("train-a", "resp-b", "resp-a", 0),
            _pair("train-a", "resp-c", "resp-e", 0),
            _pair("train-a", "resp-a", "resp-e", 0),
            _pair("train-a", "resp-b", "resp-a", 0),
        ),
        responses=(resp_a, resp_b, resp_c, resp_d, resp_e, unpaired),
        overlong_responses_removed=7,
        incomplete_pairs_removed=8,
    )
    heldout_train = TokenizedPreferenceSplit(
        pairs=(_pair("heldout-train", "heldout-better", "heldout-worse", 0),),
        responses=(
            _tokenized_response("heldout-train", "heldout-better", 8),
            _tokenized_response("heldout-train", "heldout-worse", 9),
        ),
        overlong_responses_removed=0,
        incomplete_pairs_removed=0,
    )
    heldout_test = TokenizedPreferenceSplit(
        pairs=(_pair("heldout-test", "test-better", "test-worse", 1),),
        responses=(
            _tokenized_response("heldout-test", "test-better", 10),
            _tokenized_response("heldout-test", "test-worse", 11),
        ),
        overlong_responses_removed=0,
        incomplete_pairs_removed=0,
    )
    return TokenizedPreferenceSplits(train, heldout_train, heldout_test)


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[EOS]": 2,
        **{f"token-{index}": index for index in range(3, 20)},
    }
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )


def tiny_qwen3() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=20,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
            pad_token_id=0,
            eos_token_id=2,
        )
    )


def tiny_sft_config(output_dir: Path, *, max_steps: int = 20) -> SFTTrainConfig:
    return SFTTrainConfig.from_raw(
        policy=ModelConfig.from_raw(model_id="tiny-qwen", dtype="float32", device="cpu"),
        data=DataConfig.from_raw(max_len=16),
        seed=17,
        learning_rate=0.02,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        warmup_frac=0,
        output_dir=str(output_dir),
        wandb=WandbConfig.from_raw(mode="disabled", project="sft-tests"),
    )


def test_sft_selection_is_preferred_existential_deduplicated_and_train_only() -> None:
    splits = tokenized_preference_fixture()
    dataset = build_sft_dataset(splits)
    assert [(item.source.prompt, item.source.response) for item in dataset.items] == [
        ("train-a", "resp-a"),
        ("train-a", "resp-b"),
        ("train-a", "resp-c"),
    ]
    assert dataset.counts == SFTDatasetCounts(
        train_pairs=5,
        train_unique_responses=6,
        train_overlong_responses_removed=7,
        train_incomplete_pairs_removed=8,
        sft_items=3,
        duplicate_preferred_selections_removed=2,
        dual_labeled_sft_items=1,
    )
    heldout_prompts = {
        response.source.prompt
        for split in (splits.heldout_probe_train, splits.heldout_probe_test)
        for response in split.responses
    }
    assert {item.source.prompt for item in dataset.items}.isdisjoint(heldout_prompts)
    materialized = dataset.to_huggingface()
    assert materialized.column_names == ["input_ids", "labels"]
    assert materialized["input_ids"] == [[3, 4, 5, 2], [3, 4, 6, 2], [3, 4, 7, 2]]
    assert materialized["labels"] == [
        [-100, -100, 5, 2],
        [-100, -100, 6, 2],
        [-100, -100, 7, 2],
    ]


def test_sft_selection_rejects_broken_pair_response_contracts() -> None:
    splits = tokenized_preference_fixture()
    overlapping_response = replace(
        splits.heldout_probe_train.responses[0],
        source=replace(splits.heldout_probe_train.responses[0].source, prompt="train-a"),
    )
    with pytest.raises(ValueError, match="disjoint"):
        build_sft_dataset(
            replace(
                splits,
                heldout_probe_train=replace(
                    splits.heldout_probe_train,
                    responses=(
                        overlapping_response,
                        *splits.heldout_probe_train.responses[1:],
                    ),
                ),
            )
        )
    overlapping_test_response = replace(
        splits.heldout_probe_test.responses[0],
        source=replace(splits.heldout_probe_test.responses[0].source, prompt="train-a"),
    )
    with pytest.raises(ValueError, match="disjoint"):
        build_sft_dataset(
            replace(
                splits,
                heldout_probe_test=replace(
                    splits.heldout_probe_test,
                    responses=(
                        overlapping_test_response,
                        *splits.heldout_probe_test.responses[1:],
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="better_response_id"):
        build_sft_dataset(
            replace(
                splits,
                train=replace(
                    splits.train,
                    pairs=(replace(splits.train.pairs[0], better_response_id=2),),
                ),
            )
        )
    with pytest.raises(ValueError, match="every kept pair"):
        build_sft_dataset(
            replace(splits, train=replace(splits.train, responses=splits.train.responses[:1]))
        )
    with pytest.raises(ValueError, match="unique"):
        build_sft_dataset(
            replace(
                splits,
                train=replace(
                    splits.train,
                    responses=(splits.train.responses[0], splits.train.responses[0]),
                ),
            )
        )
    with pytest.raises(ValueError, match="empty"):
        build_sft_dataset(replace(splits, train=replace(splits.train, pairs=())))


@pytest.mark.property
@given(
    lengths=st.lists(st.integers(min_value=2, max_value=8), min_size=1, max_size=5),
    left_padding=st.booleans(),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_loss_span_masking_matches_loop_reference_for_both_padding_sides(
    lengths: list[int],
    left_padding: bool,
    seed: int,
) -> None:
    width = max(lengths)
    generator = t.Generator().manual_seed(seed)
    input_ids = t.randint(1, 20, (len(lengths), width), generator=generator)
    attention_mask = t.zeros_like(input_ids, dtype=t.bool)
    spans: list[TokenSpan] = []
    expected = t.full_like(input_ids, IGNORE_INDEX)
    for row, length in enumerate(lengths):
        valid_start = width - length if left_padding else 0
        valid_stop = valid_start + length
        attention_mask[row, valid_start:valid_stop] = True
        input_ids[row, :valid_start] = 0
        input_ids[row, valid_stop:] = 0
        loss_start = valid_start + length // 2
        loss_stop = min(valid_stop, loss_start + 1 + length // 3)
        span = TokenSpan(loss_start, loss_stop)
        spans.append(span)
        for position in range(width):
            if attention_mask[row, position] and span.start <= position < span.stop:
                expected[row, position] = input_ids[row, position]
    actual = mask_loss_spans(input_ids, attention_mask, tuple(spans))
    assert t.equal(actual, expected)


def test_loss_span_masking_rejects_invalid_batch_boundaries() -> None:
    input_ids = t.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=t.int64)
    attention_mask = t.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=t.bool)
    with pytest.raises(ValueError, match="non-empty"):
        mask_loss_spans(t.empty((0, 4), dtype=t.int64), t.empty((0, 4), dtype=t.bool), ())
    with pytest.raises(ValueError, match="exactly one"):
        mask_loss_spans(input_ids, attention_mask, (TokenSpan(1, 3),))
    with pytest.raises(ValueError, match="binary"):
        mask_loss_spans(input_ids, attention_mask.to(t.int64) * 2, (TokenSpan(1, 3), TokenSpan(1, 4)))
    with pytest.raises(ValueError, match="at least one"):
        mask_loss_spans(input_ids, t.zeros_like(attention_mask), (TokenSpan(1, 3), TokenSpan(1, 4)))
    holey = attention_mask.clone()
    holey[1, 1] = False
    with pytest.raises(ValueError, match="contiguous"):
        mask_loss_spans(input_ids, holey, (TokenSpan(1, 3), TokenSpan(2, 4)))
    with pytest.raises(ValueError, match="exceeds"):
        mask_loss_spans(input_ids, attention_mask, (TokenSpan(1, 5), TokenSpan(1, 4)))
    with pytest.raises(ValueError, match="entirely"):
        mask_loss_spans(input_ids, attention_mask, (TokenSpan(2, 4), TokenSpan(1, 4)))


def test_labels_for_conversation_and_collator_right_pad_exactly() -> None:
    conversation = TokenizedConversation(
        input_ids=t.tensor([3, 4, 5], dtype=t.int64),
        attention_mask=t.ones(3, dtype=t.bool),
        loss_span=TokenSpan(1, 3),
        pool_span=TokenSpan(1, 2),
    )
    assert t.equal(labels_for_conversation(conversation), t.tensor([-100, 4, 5]))
    collated = SFTDataCollator(pad_token_id=9)(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {
                "input_ids": t.tensor([4, 5], dtype=t.int64),
                "labels": t.tensor([-100, 5], dtype=t.int64),
            },
        ]
    )
    assert t.equal(collated["input_ids"], t.tensor([[1, 2, 3], [4, 5, 9]]))
    assert t.equal(
        collated["attention_mask"],
        t.tensor([[True, True, True], [True, True, False]]),
    )
    assert t.equal(collated["labels"], t.tensor([[-100, 2, 3], [-100, 5, -100]]))


def test_collator_rejects_invalid_features_and_pad_id() -> None:
    with pytest.raises(ValueError, match="pad_token_id"):
        SFTDataCollator(-1)
    collator = SFTDataCollator(0)
    with pytest.raises(ValueError, match="empty"):
        collator([])
    with pytest.raises(ValueError, match="exactly"):
        collator([{"input_ids": [1], "labels": [1], "extra": []}])
    with pytest.raises(TypeError, match="int64"):
        collator([{"input_ids": [1.0], "labels": [1]}])
    with pytest.raises(ValueError, match="match"):
        collator([{"input_ids": [1, 2], "labels": [1]}])
    with pytest.raises(ValueError, match="equal input_ids"):
        collator([{"input_ids": [1, 2], "labels": [-100, 1]}])
    with pytest.raises(ValueError, match="loss token"):
        collator([{"input_ids": [1, 2], "labels": [-100, -100]}])
    with pytest.raises(ValueError, match="causally scoreable"):
        collator([{"input_ids": [1, 2], "labels": [1, -100]}])
    with pytest.raises(ValueError, match="rank-1"):
        collator([{"input_ids": [[1, 2]], "labels": [[1, 2]]}])
    with pytest.raises(ValueError, match="rank-1"):
        collator([{"input_ids": t.empty(0, dtype=t.int64), "labels": t.empty(0, dtype=t.int64)}])


def test_tiny_cpu_sft_decreases_loss_and_saves_exact_reload(tmp_path: Path) -> None:
    t.manual_seed(123)
    model = tiny_qwen3()
    tokenizer = tiny_tokenizer()
    output_dir = tmp_path / "tiny-sft"
    config = tiny_sft_config(output_dir)
    result = train_sft(config, model, tokenizer, tokenized_preference_fixture())
    assert len(result.loss_history) == 20
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.final_metrics["train_loss"] > 0
    assert result.dataset_counts.sft_items == 3

    manifest = json.loads((output_dir / RUN_MANIFEST_FILENAME).read_text())
    assert set(manifest) == {
        "schema_version",
        "trl_version",
        "config",
        "dataset_counts",
        "final_metrics",
        "loss_history",
    }
    assert manifest["schema_version"] == 1
    assert manifest["trl_version"] == "1.8.0"
    assert manifest["config"]["max_steps"] == 20
    assert manifest["config"]["seed"] == 17
    assert manifest["config"]["policy"]["model_id"] == "tiny-qwen"
    assert manifest["config"] == effective_config(config)
    assert manifest["dataset_counts"] == {
        "train_pairs": 5,
        "train_unique_responses": 6,
        "train_overlong_responses_removed": 7,
        "train_incomplete_pairs_removed": 8,
        "sft_items": 3,
        "duplicate_preferred_selections_removed": 2,
        "dual_labeled_sft_items": 1,
    }
    assert math.isfinite(manifest["final_metrics"]["train_loss"])
    assert manifest["loss_history"] == list(result.loss_history)
    assert json.loads((output_dir / EFFECTIVE_CONFIG_FILENAME).read_text()) == effective_config(config)

    log_records = [
        json.loads(line)
        for line in (output_dir / TRAIN_LOGS_FILENAME).read_text().splitlines()
    ]
    assert [record["loss"] for record in log_records if "loss" in record] == list(
        result.loss_history
    )
    assert any("train_loss" in record for record in log_records)

    reloaded = AutoModelForCausalLM.from_pretrained(output_dir)
    for name, value in model.state_dict().items():
        assert t.equal(value.cpu(), reloaded.state_dict()[name].cpu())
    reloaded_tokenizer = cast(Any, AutoTokenizer.from_pretrained(output_dir))
    assert reloaded_tokenizer.pad_token_id == tokenizer.pad_token_id


def test_sft_refuses_nonempty_or_nondirectory_output_paths(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "marker").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="refusing"):
        train_sft(
            tiny_sft_config(occupied, max_steps=1),
            tiny_qwen3(),
            tiny_tokenizer(),
            tokenized_preference_fixture(),
        )
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("file")
    with pytest.raises(NotADirectoryError):
        train_sft(
            tiny_sft_config(file_path, max_steps=1),
            tiny_qwen3(),
            tiny_tokenizer(),
            tokenized_preference_fixture(),
        )


def test_same_seed_produces_identical_first_step_loss(tmp_path: Path) -> None:
    t.manual_seed(991)
    initial = {name: value.clone() for name, value in tiny_qwen3().state_dict().items()}
    first_losses: list[float] = []
    for run_index in range(2):
        model = tiny_qwen3()
        model.load_state_dict(initial)
        result = train_sft(
            tiny_sft_config(tmp_path / f"deterministic-{run_index}", max_steps=1),
            model,
            tiny_tokenizer(),
            tokenized_preference_fixture(),
        )
        first_losses.append(result.loss_history[0])
    assert first_losses[0] == first_losses[1]


def test_warmup_steps_mapping_is_exact_and_integer(tmp_path: Path) -> None:
    def config_for(max_steps: int, warmup_frac: float) -> SFTTrainConfig:
        return replace(
            tiny_sft_config(tmp_path / "unused", max_steps=max_steps),
            warmup_frac=parse_unit_interval(warmup_frac),
        )

    assert warmup_steps_for(config_for(1000, 0.1)) == 100
    assert warmup_steps_for(config_for(1000, 0.999)) == 999
    assert warmup_steps_for(config_for(1000, 1.0)) == 1000
    assert warmup_steps_for(config_for(20, 0.0)) == 0
    assert warmup_steps_for(config_for(3, 0.5)) == 2


def test_train_sft_seeds_before_any_training_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    dummy = SFTResult(
        output_dir=tmp_path,
        dataset_counts=SFTDatasetCounts(0, 0, 0, 0, 0, 0, 0),
        final_metrics={},
        loss_history=(1.0,),
    )
    monkeypatch.setattr(
        sft_module, "seed_all", lambda seed: calls.append(f"seed_all:{int(seed)}")
    )

    def fake_train(*_args: object) -> SFTResult:
        calls.append("_train_sft")
        return dummy

    monkeypatch.setattr(sft_module, "_train_sft", fake_train)
    train_sft(
        tiny_sft_config(tmp_path / "seed-order"),
        tiny_qwen3(),
        tiny_tokenizer(),
        tokenized_preference_fixture(),
    )
    assert calls == ["seed_all:17", "_train_sft"]


def test_sft_crashes_on_nonfinite_loss_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NanTrainer:
        def __init__(self, **_kwargs: object) -> None:
            self.state = SimpleNamespace(log_history=[{"loss": float("nan")}])
            self.model = None
            self.accelerator = SimpleNamespace(unwrap_model=lambda model: model)

        def train(self) -> SimpleNamespace:
            return SimpleNamespace(metrics={"train_loss": float("nan")})

    monkeypatch.setattr(sft_module, "SFTTrainer", NanTrainer)
    with pytest.raises(FloatingPointError, match="finite"):
        train_sft(
            tiny_sft_config(tmp_path / "nan-run", max_steps=1),
            tiny_qwen3(),
            tiny_tokenizer(),
            tokenized_preference_fixture(),
        )


def test_train_rejects_tokenizer_without_pad_token(tmp_path: Path) -> None:
    vocabulary = {
        "[UNK]": 1,
        "[EOS]": 2,
        **{f"token-{index}": index for index in range(3, 20)},
    }
    padless = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    with pytest.raises(ValueError, match="pad_token_id"):
        train_sft(
            tiny_sft_config(tmp_path / "no-pad", max_steps=1),
            tiny_qwen3(),
            padless,
            tokenized_preference_fixture(),
        )


def _one_pair_subset(split: PreferenceSplit) -> PreferenceSplit:
    pair = split.pairs[0]
    keys = {(pair.prompt, pair.response_0), (pair.prompt, pair.response_1)}
    responses = tuple(response for response in split.responses if response.key in keys)
    if len(responses) != 2:
        raise AssertionError("live pair must resolve to two deduplicated responses")
    return PreferenceSplit((pair,), responses, 0, 0, 0)


@pytest.mark.slow
def test_real_qwen3_few_step_sft_checkpoint_and_ppl_smoke(tmp_path: Path) -> None:
    device = "cuda" if t.cuda.is_available() else "cpu"
    dtype = "bfloat16" if t.cuda.is_available() else "float32"
    policy_config = ModelConfig.from_raw(
        model_id=DEV_POLICY_MODEL_ID,
        dtype=dtype,
        device=device,
    )
    loaded = load_policy(policy_config)
    data_config = DataConfig.from_raw(max_len=256, seed=0)
    raw = load_preference_dataset()
    full_splits = build_prompt_splits(raw, data_config)
    small_splits = PreferenceSplits(
        _one_pair_subset(full_splits.train),
        _one_pair_subset(full_splits.heldout_probe_train),
        _one_pair_subset(full_splits.heldout_probe_test),
    )
    tokenized = tokenize_preference_splits(small_splits, loaded.tokenizer, data_config)
    config = SFTTrainConfig.from_raw(
        policy=policy_config,
        data=data_config,
        learning_rate=5.0e-5,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=3,
        warmup_frac=0,
        output_dir=str(tmp_path / "real-sft"),
        wandb=WandbConfig.from_raw(mode="disabled"),
    )
    result = train_sft(config, loaded.model, loaded.tokenizer, tokenized)
    assert all(t.isfinite(t.tensor(result.loss_history)))
    assert result.loss_history[-1] < result.loss_history[0]

    collated = SFTDataCollator(cast(int, loaded.tokenizer.pad_token_id))(
        [
            {
                "input_ids": response.conversation.input_ids,
                "labels": labels_for_conversation(response.conversation),
            }
            for response in tokenized.heldout_probe_train.responses
        ]
    )
    ppl = heldout_response_perplexity(
        loaded.model,
        collated["input_ids"].to(device),
        collated["attention_mask"].to(device),
        collated["labels"].to(device),
    )
    assert bool(t.isfinite(ppl.perplexity))

    first_name, first_value = next(iter(loaded.model.state_dict().items()))
    expected = first_value.detach().cpu().clone()
    del loaded
    gc.collect()
    t.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(
        tmp_path / "real-sft",
        dtype=t.bfloat16 if dtype == "bfloat16" else t.float32,
    )
    assert t.equal(expected, reloaded.state_dict()[first_name].cpu())
