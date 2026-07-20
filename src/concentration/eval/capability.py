"""Held-out response language-model capability evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Bool, Float32, Int, Int64, jaxtyped

BatchTokenIds = Int64[t.Tensor, "batch tokens"]
BatchTokenMask = Bool[t.Tensor, "batch tokens"] | Int[t.Tensor, "batch tokens"]
ScalarFloat32 = Float32[t.Tensor, ""]
IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class ResponsePerplexity:
    """Token-summed and token-mean fp32 NLL plus its exact exponential."""

    nll_sum: ScalarFloat32
    mean_nll: ScalarFloat32
    perplexity: ScalarFloat32
    token_count: int


@jaxtyped(typechecker=beartype)
def heldout_response_perplexity(
    model: t.nn.Module,
    input_ids: BatchTokenIds,
    attention_mask: BatchTokenMask,
    labels: BatchTokenIds,
) -> ResponsePerplexity:
    """Compute masked response-token NLL and perplexity with one policy forward.

    ``labels`` must be the Phase 1 ``loss_span`` mask: token IDs at response-loss
    positions and ``-100`` at prompt and padding positions. Causal shifting is applied
    here, so each selected label is scored by the preceding policy position.
    """
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("perplexity requires a non-empty batch and sequence")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("attention_mask values must be binary")
    valid = attention_mask.bool()
    if not bool(valid.any(dim=1).all()):
        raise ValueError("every perplexity item must contain a non-padding token")
    selected_labels = labels.ne(IGNORE_INDEX)
    if not bool(selected_labels.any(dim=1).all()):
        raise ValueError("every perplexity item must contain at least one loss token")
    if bool((selected_labels & ~valid).any()):
        raise ValueError("padding labels must be -100")
    if bool((selected_labels & labels.ne(input_ids)).any()):
        raise ValueError("unmasked labels must equal input_ids")

    model.eval()
    with t.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    logits = outputs.logits
    if not isinstance(logits, t.Tensor) or logits.ndim != 3:
        raise TypeError("causal LM must return rank-3 logits")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("causal LM logits must match the input batch and sequence dimensions")
    shifted_labels = labels[:, 1:].contiguous()
    if not bool(shifted_labels.ne(IGNORE_INDEX).any(dim=1).all()):
        raise ValueError("every perplexity item must contain a causally scoreable loss token")
    token_count = int(shifted_labels.ne(IGNORE_INDEX).sum())
    shifted_logits = logits[:, :-1, :].float().contiguous()
    if not bool(t.isfinite(shifted_logits).all()):
        raise ValueError("causal LM logits must be finite")
    nll_sum = t.nn.functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    mean_nll = nll_sum / token_count
    return ResponsePerplexity(
        nll_sum=nll_sum,
        mean_nll=mean_nll,
        perplexity=mean_nll.exp(),
        token_count=token_count,
    )
