"""Qwen3 policy loading, layer hooks, and mask-aware response pooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch as t
from beartype import beartype
from jaxtyping import Bool, Float, Float32, Int, Int64, jaxtyped
from transformers import AutoModelForCausalLM, AutoTokenizer

from concentration.config import ModelConfig, Pooling, RepExtractionConfig, torch_dtype
from concentration.types import PooledRepresentations

TokenIds = Int64[t.Tensor, "batch tokens"]
TokenMask = Int[t.Tensor, "batch tokens"] | Bool[t.Tensor, "batch tokens"]
HiddenStates = Float[t.Tensor, "batch tokens hidden"]
Float32HiddenStates = Float32[t.Tensor, "batch tokens hidden"]
Logits = Float[t.Tensor, "batch tokens vocabulary"]


@dataclass(frozen=True, slots=True)
class LoadedPolicy:
    """A causal language model and its matching tokenizer."""

    model: t.nn.Module
    tokenizer: Any


@dataclass(frozen=True, slots=True)
class LayerForward:
    """Layer-l hidden states and logits produced by one policy forward pass."""

    hidden_states: Float32HiddenStates
    logits: Logits


@dataclass(frozen=True, slots=True)
class ExtractedPolicyOutput:
    """Pooled fp32 representations and same-forward policy logits."""

    representations: PooledRepresentations
    logits: Logits


@beartype
def load_policy(config: ModelConfig) -> LoadedPolicy:
    """Load the configured causal LM and tokenizer from their exact revision."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
    model = cast(
        t.nn.Module,
        AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            dtype=torch_dtype(config.dtype),
        ),
    )
    model.to(config.device)
    return LoadedPolicy(model=model, tokenizer=tokenizer)


def _decoder_layers(model: t.nn.Module) -> t.nn.ModuleList:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("policy must expose Qwen3 decoder blocks as model.layers")
    layers = model.model.layers
    if not isinstance(layers, t.nn.ModuleList):
        raise TypeError("policy model.layers must be torch.nn.ModuleList")
    return layers


@jaxtyped(typechecker=beartype)
def forward_at_layer(
    model: t.nn.Module,
    input_ids: TokenIds,
    attention_mask: TokenMask,
    layer: int,
) -> LayerForward:
    """Capture the input to decoder block ``layer`` and logits from one forward."""
    layers = _decoder_layers(model)
    if type(layer) is not int or not 0 <= layer < len(layers):
        raise ValueError(f"layer must be in [0, {len(layers) - 1}]")
    captured: list[t.Tensor] = []

    def capture_hidden(_module: t.nn.Module, args: tuple[object, ...]) -> None:
        if not args or not isinstance(args[0], t.Tensor):
            raise TypeError("decoder pre-hook did not receive hidden states")
        captured.append(args[0])

    handle = layers[layer].register_forward_pre_hook(capture_hidden)
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"decoder layer hook fired {len(captured)} times instead of once")
    logits = outputs.logits
    if not isinstance(logits, t.Tensor) or logits.ndim != 3:
        raise TypeError("causal LM must return rank-3 logits")
    hidden_states = captured[0]
    if hidden_states.ndim != 3 or hidden_states.shape[:2] != input_ids.shape:
        raise ValueError("hooked hidden-state shape does not match input tokens")
    return LayerForward(hidden_states=hidden_states.float(), logits=logits)


@jaxtyped(typechecker=beartype)
def pool_hidden_states(
    hidden_states: HiddenStates,
    pool_mask: TokenMask,
    pooling: Pooling,
) -> PooledRepresentations:
    """Pool selected response tokens in fp32, independent of padding side."""
    mask = pool_mask.bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every item must select at least one response token")
    values = hidden_states.float()
    expanded_mask = mask.unsqueeze(-1)
    if pooling is Pooling.MEAN:
        pooled = values.masked_fill(~expanded_mask, 0.0).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        )
    elif pooling is Pooling.LAST:
        positions = t.arange(values.shape[1], device=values.device).expand_as(mask)
        last_indices = positions.masked_fill(~mask, -1).max(dim=1).values
        batch_indices = t.arange(values.shape[0], device=values.device)
        pooled = values[batch_indices, last_indices]
    elif pooling is Pooling.MAX:
        pooled = values.masked_fill(~expanded_mask, -t.inf).max(dim=1).values
    elif pooling is Pooling.MIN:
        pooled = values.masked_fill(~expanded_mask, t.inf).min(dim=1).values
    return PooledRepresentations.from_tensor(pooled)


@jaxtyped(typechecker=beartype)
def extract_policy_output(
    model: t.nn.Module,
    input_ids: TokenIds,
    attention_mask: TokenMask,
    pool_mask: TokenMask,
    config: RepExtractionConfig,
) -> ExtractedPolicyOutput:
    """Return pooled layer representations and logits from a single model call."""
    forwarded = forward_at_layer(model, input_ids, attention_mask, config.layer)
    pooled = pool_hidden_states(forwarded.hidden_states, pool_mask, config.pooling)
    return ExtractedPolicyOutput(representations=pooled, logits=forwarded.logits)
