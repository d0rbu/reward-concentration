from __future__ import annotations

import gc
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch as t
from hypothesis import given
from hypothesis import strategies as st
from jaxtyping import TypeCheckError
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from concentration.config import ModelConfig, ModelDType, Pooling, RepExtractionConfig
from concentration.models.policy import (
    extract_policy_output,
    forward_at_layer,
    load_policy,
    pool_hidden_states,
)


def tiny_qwen3() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    return Qwen3ForCausalLM(config).eval()


def test_hooked_layer_exactly_equals_output_hidden_states_index() -> None:
    model = tiny_qwen3()
    input_ids = t.tensor([[1, 2, 3, 4], [0, 0, 5, 6]], dtype=t.int64)
    attention_mask = input_ids.ne(0)
    for layer in range(model.config.num_hidden_layers):
        hooked = forward_at_layer(model, input_ids, attention_mask, layer)
        reference = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        assert t.equal(hooked.hidden_states, reference.hidden_states[layer].float())
        assert t.equal(hooked.logits, reference.logits)


def test_layer_extraction_uses_one_forward_and_does_not_request_hidden_states() -> None:
    model = tiny_qwen3()
    original_forward = model.forward
    calls: list[dict[str, object]] = []

    def counted_forward(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return original_forward(*args, **kwargs)

    model.forward = counted_forward  # type: ignore[method-assign]
    input_ids = t.tensor([[1, 2, 3]], dtype=t.int64)
    output = forward_at_layer(model, input_ids, t.ones_like(input_ids), 1)
    assert output.hidden_states.dtype == t.float32
    assert len(calls) == 1
    assert "output_hidden_states" not in calls[0]


@pytest.mark.parametrize("layer", [-1, 3, True])
def test_layer_extraction_rejects_invalid_layer(layer: int) -> None:
    input_ids = t.tensor([[1, 2]], dtype=t.int64)
    with pytest.raises(ValueError, match="layer"):
        forward_at_layer(tiny_qwen3(), input_ids, t.ones_like(input_ids), layer)


def test_layer_extraction_rejects_non_qwen_structure_and_wrong_shapes() -> None:
    input_ids = t.tensor([[1, 2]], dtype=t.int64)
    with pytest.raises(TypeError, match="model.layers"):
        forward_at_layer(nn.Linear(2, 2), input_ids, t.ones_like(input_ids), 0)
    with pytest.raises(TypeCheckError):
        forward_at_layer(tiny_qwen3(), input_ids, t.ones(3, dtype=t.int64), 0)


class SyntheticHookPolicy(nn.Module):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure
        self.model = SyntheticBackbone()

    def forward(self, *, input_ids: t.Tensor, **_kwargs: object) -> SimpleNamespace:
        if self.failure == "no_hook":
            hidden = t.ones((*input_ids.shape, 4))
        elif self.failure == "bad_hook_args":
            hidden = self.model.layers[0](input=t.ones((*input_ids.shape, 4)))
        elif self.failure == "bad_hidden":
            hidden = self.model.layers[0](t.ones((input_ids.shape[0], input_ids.shape[1] + 1, 4)))
        else:
            hidden = self.model.layers[0](t.ones((*input_ids.shape, 4)))
        logits = t.ones((*input_ids.shape, 5)) if self.failure != "bad_logits" else t.ones(input_ids.shape)
        return SimpleNamespace(logits=logits, hidden=hidden)


class SyntheticBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])


def test_layer_extraction_crashes_on_malformed_model_contracts() -> None:
    input_ids = t.tensor([[1, 2]], dtype=t.int64)
    mask = t.ones_like(input_ids)
    not_module_list = cast(Any, nn.Module())
    not_module_list.model = SimpleNamespace(layers=[])
    with pytest.raises(TypeError, match="ModuleList"):
        forward_at_layer(not_module_list, input_ids, mask, 0)
    with pytest.raises(RuntimeError, match="0 times"):
        forward_at_layer(SyntheticHookPolicy("no_hook"), input_ids, mask, 0)
    with pytest.raises(TypeError, match="pre-hook"):
        forward_at_layer(SyntheticHookPolicy("bad_hook_args"), input_ids, mask, 0)
    with pytest.raises(TypeError, match="rank-3 logits"):
        forward_at_layer(SyntheticHookPolicy("bad_logits"), input_ids, mask, 0)
    with pytest.raises(ValueError, match="shape"):
        forward_at_layer(SyntheticHookPolicy("bad_hidden"), input_ids, mask, 0)


def _loop_pool(hidden: t.Tensor, mask: t.Tensor, pooling: Pooling) -> t.Tensor:
    rows: list[t.Tensor] = []
    for batch_index in range(hidden.shape[0]):
        selected = hidden[batch_index][mask[batch_index].bool()].float()
        if pooling is Pooling.MEAN:
            rows.append(selected.sum(dim=0) / selected.shape[0])
        elif pooling is Pooling.LAST:
            rows.append(selected[-1])
        elif pooling is Pooling.MAX:
            values = [max(float(row[column]) for row in selected) for column in range(hidden.shape[2])]
            rows.append(t.tensor(values, dtype=t.float32))
        else:
            values = [min(float(row[column]) for row in selected) for column in range(hidden.shape[2])]
            rows.append(t.tensor(values, dtype=t.float32))
    return t.stack(rows)


@pytest.mark.property
@given(
    batch=st.integers(min_value=1, max_value=5),
    tokens=st.integers(min_value=1, max_value=8),
    hidden_size=st.integers(min_value=1, max_value=7),
    left_padding=st.booleans(),
    seed=st.integers(min_value=0, max_value=1000),
)
@pytest.mark.parametrize("pooling", list(Pooling))
def test_pooling_matches_loop_reference_for_random_shapes_and_padding(
    batch: int,
    tokens: int,
    hidden_size: int,
    left_padding: bool,
    seed: int,
    pooling: Pooling,
) -> None:
    generator = t.Generator().manual_seed(seed)
    hidden = t.randn(batch, tokens, hidden_size, generator=generator, dtype=t.float64)
    lengths = t.randint(1, tokens + 1, (batch,), generator=generator)
    positions = t.arange(tokens).expand(batch, tokens)
    mask = (
        positions >= tokens - lengths[:, None]
        if left_padding
        else positions < lengths[:, None]
    )
    actual = pool_hidden_states(hidden, mask, pooling).tensor
    expected = _loop_pool(hidden, mask, pooling)
    assert actual.dtype == t.float32
    assert t.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_pooling_known_values() -> None:
    hidden = t.tensor(
        [
            [[100.0, 100.0], [1.0, 4.0], [3.0, 2.0]],
            [[5.0, 1.0], [2.0, 8.0], [100.0, 100.0]],
        ],
        dtype=t.float32,
    )
    mask = t.tensor([[0, 1, 1], [1, 1, 0]])
    assert t.equal(pool_hidden_states(hidden, mask, Pooling.MEAN).tensor, t.tensor([[2.0, 3.0], [3.5, 4.5]]))
    assert t.equal(pool_hidden_states(hidden, mask, Pooling.LAST).tensor, t.tensor([[3.0, 2.0], [2.0, 8.0]]))
    assert t.equal(pool_hidden_states(hidden, mask, Pooling.MAX).tensor, t.tensor([[3.0, 4.0], [5.0, 8.0]]))
    assert t.equal(pool_hidden_states(hidden, mask, Pooling.MIN).tensor, t.tensor([[1.0, 2.0], [2.0, 1.0]]))


def test_pooling_rejects_empty_rows_shape_mismatch_and_unrefined_mode() -> None:
    hidden = t.ones((2, 3, 4), dtype=t.float32)
    with pytest.raises(ValueError, match="at least one"):
        pool_hidden_states(hidden, t.tensor([[1, 0, 0], [0, 0, 0]]), Pooling.MEAN)
    with pytest.raises(TypeCheckError):
        pool_hidden_states(hidden, t.ones((2, 2), dtype=t.bool), Pooling.MEAN)
    with pytest.raises(TypeCheckError):
        pool_hidden_states(
            hidden,
            t.ones((2, 3), dtype=t.bool),
            "median",  # ty: ignore[invalid-argument-type]
        )


def test_extract_policy_output_returns_pool_and_logits_from_one_call() -> None:
    model = tiny_qwen3()
    counter = Mock()

    def count_forward(*_args: object) -> None:
        counter()

    handle = model.register_forward_pre_hook(count_forward)
    input_ids = t.tensor([[1, 2, 3], [4, 5, 6]], dtype=t.int64)
    attention_mask = t.ones_like(input_ids)
    pool_mask = t.tensor([[0, 1, 1], [0, 0, 1]], dtype=t.bool)
    output = extract_policy_output(
        model,
        input_ids,
        attention_mask,
        pool_mask,
        RepExtractionConfig.from_raw(layer=1, pooling="mean"),
    )
    handle.remove()
    assert counter.call_count == 1
    assert output.representations.tensor.shape == (2, 16)
    assert output.logits.shape == (2, 3, 41)


def test_load_policy_uses_exact_config(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = object()
    model = Mock()
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(
        "concentration.models.policy.AutoTokenizer.from_pretrained",
        tokenizer_loader,
    )
    monkeypatch.setattr(
        "concentration.models.policy.AutoModelForCausalLM.from_pretrained",
        model_loader,
    )
    config = ModelConfig.from_raw(
        model_id="org/model",
        revision="commit",
        dtype="float32",
        device="cpu",
    )
    loaded = load_policy(config)
    tokenizer_loader.assert_called_once_with("org/model", revision="commit")
    model_loader.assert_called_once_with(
        "org/model",
        revision="commit",
        dtype=t.float32,
    )
    model.to.assert_called_once_with("cpu")
    assert loaded.model is model
    assert loaded.tokenizer is tokenizer


def test_model_dtype_mapping_is_exhaustive() -> None:
    assert ModelConfig.from_raw(dtype=ModelDType.FLOAT32.value).dtype is ModelDType.FLOAT32


@pytest.mark.slow
def test_real_qwen3_policy_loads_and_extracts() -> None:
    loaded = load_policy(ModelConfig.from_raw(dtype="float32", device="cpu"))
    loaded.model.eval()
    tokenizer = loaded.tokenizer
    batch = tokenizer("A tiny policy smoke test.", return_tensors="pt")
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    output = forward_at_layer(loaded.model, input_ids, attention_mask, 1)
    assert output.hidden_states.dtype == t.float32
    assert output.hidden_states.shape[:2] == input_ids.shape
    assert output.logits.shape[:2] == input_ids.shape
    assert bool(t.isfinite(output.hidden_states).all())
    del output, batch, input_ids, attention_mask, loaded
    gc.collect()
