from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

import concentration.cli as cli_module
import concentration.tracka.sft as sft_module
from concentration.cli import load_sft_config, main, parse_sft_config, run_ppl
from concentration.config import DataConfig, ModelConfig, SFTTrainConfig, WandbConfig
from concentration.data.preference import (
    PreferencePair,
    PreferenceResponse,
    TokenizedConversation,
    TokenizedPreferenceResponse,
    TokenizedPreferenceSplit,
    TokenizedPreferenceSplits,
    TokenSpan,
)
from concentration.models.policy import LoadedPolicy
from concentration.tracka.sft import RUN_MANIFEST_FILENAME
from concentration.types import parse_rank


def _digest(prompt: str, response: str) -> str:
    return hashlib.sha256((prompt + response).encode()).hexdigest()


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[EOS]": 2,
        **{f"token-{index}": index for index in range(3, 12)},
    }
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )


def _tiny_model() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=12,
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


def _tokenized_split(
    prompt: str,
    token: int,
    *,
    response_tokens: int = 1,
) -> TokenizedPreferenceSplit:
    first = f"{prompt}-first"
    second = f"{prompt}-second"
    pair = PreferencePair(
        prompt,
        first,
        second,
        _digest(prompt, first),
        _digest(prompt, second),
        0,
    )
    responses: list[TokenizedPreferenceResponse] = []
    for offset, text in enumerate((first, second)):
        source = PreferenceResponse(prompt, text, _digest(prompt, text))
        content = [token + offset + 2 * extra for extra in range(response_tokens)]
        conversation = TokenizedConversation(
            t.tensor([3, 4, *content, 2], dtype=t.int64),
            t.ones(3 + len(content), dtype=t.bool),
            TokenSpan(2, 3 + len(content)),
            TokenSpan(2, 2 + len(content)),
        )
        responses.append(TokenizedPreferenceResponse(source, conversation))
    return TokenizedPreferenceSplit((pair,), tuple(responses), 0, 0)


def _tokenized_splits() -> TokenizedPreferenceSplits:
    """heldout_probe_train has 3 causally scoreable tokens per response; train has 2.

    The asymmetry makes ppl's token_count identify which split was evaluated.
    """
    return TokenizedPreferenceSplits(
        _tokenized_split("train", 5),
        _tokenized_split("heldout-train", 7, response_tokens=2),
        _tokenized_split("heldout-test", 9),
    )


def _write_tiny_config(path: Path, output_dir: Path, *, max_steps: int = 3) -> None:
    path.write_text(
        "\n".join(
            (
                "[policy]",
                'model_id = "tiny-qwen"',
                'revision = "local"',
                'dtype = "float32"',
                'device = "cpu"',
                "",
                "[data]",
                "max_len = 16",
                "heldout_probe_train_frac = 0.2",
                "heldout_probe_test_frac = 0.2",
                "seed = 3",
                "",
                "[sft]",
                "seed = 7",
                "learning_rate = 0.02",
                "per_device_batch_size = 1",
                "gradient_accumulation_steps = 1",
                f"max_steps = {max_steps}",
                "warmup_frac = 0.0",
                f"output_dir = {json.dumps(str(output_dir))}",
                "",
                "[wandb]",
                'mode = "disabled"',
                'project = "cli-test"',
                "",
            )
        )
    )


def test_toml_parse_refines_to_exact_config(tmp_path: Path) -> None:
    config_path = tmp_path / "sft.toml"
    output_dir = tmp_path / "output"
    _write_tiny_config(config_path, output_dir, max_steps=11)
    actual = load_sft_config(config_path)
    expected = SFTTrainConfig.from_raw(
        policy=ModelConfig.from_raw(
            model_id="tiny-qwen",
            revision="local",
            dtype="float32",
            device="cpu",
        ),
        data=DataConfig.from_raw(
            max_len=16,
            heldout_probe_train_frac=0.2,
            heldout_probe_test_frac=0.2,
            seed=3,
        ),
        seed=7,
        learning_rate=0.02,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=11,
        warmup_frac=0,
        output_dir=str(output_dir),
        wandb=WandbConfig.from_raw(mode="disabled", project="cli-test"),
    )
    assert actual == expected


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": {}},
        {"policy": {"unknown": 1}},
        {"data": {"unknown": 1}},
        {"sft": {"epochs": 1}},
        {"wandb": {"unknown": 1}},
    ],
)
def test_toml_parser_rejects_unknown_keys(raw: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_sft_config(raw)


def test_toml_parser_rejects_non_table_sections() -> None:
    with pytest.raises(TypeError, match="table"):
        parse_sft_config({"sft": cast(Any, "not-a-table")})


def test_sft_cli_runs_tiny_cpu_training_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "cli-sft"
    config_path = tmp_path / "sft.toml"
    _write_tiny_config(config_path, output_dir)
    loaded = LoadedPolicy(_tiny_model(), _tiny_tokenizer())
    monkeypatch.setattr(sft_module, "load_policy", lambda _config: loaded)
    monkeypatch.setattr(sft_module, "load_preference_dataset", lambda _dataset_id: object())
    monkeypatch.setattr(sft_module, "build_prompt_splits", lambda _raw, _config: object())
    monkeypatch.setattr(
        sft_module,
        "tokenize_preference_splits",
        lambda _splits, _tokenizer, _config: _tokenized_splits(),
    )
    assert main(["sft", str(config_path)]) == 0
    manifest = json.loads((output_dir / RUN_MANIFEST_FILENAME).read_text())
    assert manifest["dataset_counts"]["sft_items"] == 1
    assert len(manifest["loss_history"]) == 3


def test_ppl_cli_evaluates_checkpoint_with_count_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "ppl.toml"
    _write_tiny_config(config_path, tmp_path / "unused")
    loaded = LoadedPolicy(_tiny_model(), _tiny_tokenizer())
    monkeypatch.setattr(cli_module, "load_policy", lambda _config: loaded)
    monkeypatch.setattr(cli_module, "load_preference_dataset", lambda _dataset_id: object())
    monkeypatch.setattr(cli_module, "build_prompt_splits", lambda _raw, _config: object())
    monkeypatch.setattr(
        cli_module,
        "tokenize_preference_splits",
        lambda _splits, _tokenizer, _config: _tokenized_splits(),
    )
    assert main(["ppl", str(config_path), "checkpoint", "--count", "1", "--batch-size", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint"] == "checkpoint"
    assert payload["evaluated_responses"] == 1
    assert payload["token_count"] == 3
    assert payload["perplexity"] > 0

    assert main(["ppl", str(config_path), "checkpoint", "--batch-size", "1"]) == 0
    payload_all = json.loads(capsys.readouterr().out)
    assert payload_all["evaluated_responses"] == 2
    assert payload_all["token_count"] == 6
    assert payload_all["perplexity"] == pytest.approx(math.exp(payload_all["mean_nll"]))


def test_ppl_runner_rejects_blank_checkpoint_before_loading() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        run_ppl(
            SFTTrainConfig.from_raw(wandb=WandbConfig.from_raw(mode="disabled")),
            " ",
            None,
            parse_rank(1),
        )
