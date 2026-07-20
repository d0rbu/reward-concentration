from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t
from hypothesis import given
from hypothesis import strategies as st

from concentration.eval.capability import IGNORE_INDEX, heldout_response_perplexity


class FixedLogitPolicy(t.nn.Module):
    def __init__(self, logits: t.Tensor | object) -> None:
        super().__init__()
        self.logits = logits
        self.calls = 0

    def forward(self, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(logits=self.logits)


def _loop_nll(logits: t.Tensor, labels: t.Tensor) -> tuple[t.Tensor, int]:
    total = t.zeros((), dtype=t.float32)
    count = 0
    log_probs = logits.float().log_softmax(dim=-1)
    for batch_index in range(labels.shape[0]):
        for token_index in range(1, labels.shape[1]):
            target = int(labels[batch_index, token_index])
            if target != IGNORE_INDEX:
                total -= log_probs[batch_index, token_index - 1, target]
                count += 1
    return total, count


def test_heldout_response_perplexity_matches_hand_loop_with_one_forward() -> None:
    logits = t.tensor(
        [
            [[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 4.0], [0.0, 0.0, 0.0]],
            [[1.0, 2.0, 0.0], [3.0, 0.0, 1.0], [0.0, 4.0, 1.0], [2.0, 0.0, 3.0]],
        ],
        dtype=t.float64,
    )
    input_ids = t.tensor([[0, 1, 2, 0], [2, 0, 1, 2]], dtype=t.int64)
    attention_mask = t.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=t.bool)
    labels = t.tensor(
        [[IGNORE_INDEX, 1, 2, IGNORE_INDEX], [IGNORE_INDEX, IGNORE_INDEX, 1, 2]],
        dtype=t.int64,
    )
    model = FixedLogitPolicy(logits)
    result = heldout_response_perplexity(model, input_ids, attention_mask, labels)
    expected_sum, expected_count = _loop_nll(logits, labels)
    assert model.calls == 1
    assert result.nll_sum.dtype == t.float32
    assert t.allclose(result.nll_sum, expected_sum, atol=1.0e-6, rtol=1.0e-6)
    assert result.token_count == expected_count
    assert t.allclose(result.mean_nll, expected_sum / expected_count)
    assert t.equal(result.perplexity, result.mean_nll.exp())


@pytest.mark.property
@given(
    batch=st.integers(min_value=1, max_value=4),
    tokens=st.integers(min_value=2, max_value=7),
    vocabulary=st.integers(min_value=2, max_value=9),
    left_padding=st.booleans(),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_perplexity_matches_loop_over_random_shapes_and_padding(
    batch: int,
    tokens: int,
    vocabulary: int,
    left_padding: bool,
    seed: int,
) -> None:
    generator = t.Generator().manual_seed(seed)
    lengths = t.randint(2, tokens + 1, (batch,), generator=generator)
    input_ids = t.randint(0, vocabulary, (batch, tokens), generator=generator)
    attention_mask = t.zeros((batch, tokens), dtype=t.bool)
    labels = t.full_like(input_ids, IGNORE_INDEX)
    for row, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        valid_start = tokens - length if left_padding else 0
        valid_stop = valid_start + length
        response_start = valid_start + length // 2
        attention_mask[row, valid_start:valid_stop] = True
        input_ids[row, :valid_start] = 0
        input_ids[row, valid_stop:] = 0
        labels[row, response_start:valid_stop] = input_ids[row, response_start:valid_stop]
    logits = t.randn(batch, tokens, vocabulary, generator=generator, dtype=t.float64)
    result = heldout_response_perplexity(
        FixedLogitPolicy(logits),
        input_ids,
        attention_mask,
        labels,
    )
    expected_sum, expected_count = _loop_nll(logits, labels)
    assert t.allclose(result.nll_sum, expected_sum, atol=2.0e-6, rtol=2.0e-6)
    assert result.token_count == expected_count
    assert t.allclose(result.mean_nll, expected_sum / expected_count, atol=1.0e-6)
    assert t.equal(result.perplexity, result.mean_nll.exp())


def test_perplexity_rejects_empty_or_invalid_masks_and_labels() -> None:
    logits = t.zeros((1, 3, 4))
    input_ids = t.tensor([[1, 2, 3]], dtype=t.int64)
    attention_mask = t.ones_like(input_ids, dtype=t.bool)
    labels = t.tensor([[IGNORE_INDEX, 2, 3]], dtype=t.int64)
    with pytest.raises(ValueError, match="non-empty"):
        heldout_response_perplexity(
            FixedLogitPolicy(t.empty((0, 3, 4))),
            t.empty((0, 3), dtype=t.int64),
            t.empty((0, 3), dtype=t.bool),
            t.empty((0, 3), dtype=t.int64),
        )
    with pytest.raises(ValueError, match="binary"):
        heldout_response_perplexity(
            FixedLogitPolicy(logits),
            input_ids,
            attention_mask.to(t.int64) * 2,
            labels,
        )
    with pytest.raises(ValueError, match="non-padding"):
        heldout_response_perplexity(
            FixedLogitPolicy(logits),
            input_ids,
            t.zeros_like(attention_mask),
            t.full_like(labels, IGNORE_INDEX),
        )
    padded_mask = t.tensor([[1, 1, 0]], dtype=t.bool)
    with pytest.raises(ValueError, match="padding labels"):
        heldout_response_perplexity(
            FixedLogitPolicy(logits),
            input_ids,
            padded_mask,
            labels,
        )
    with pytest.raises(ValueError, match="equal input_ids"):
        heldout_response_perplexity(
            FixedLogitPolicy(logits),
            input_ids,
            attention_mask,
            t.tensor([[IGNORE_INDEX, 3, 3]], dtype=t.int64),
        )
    with pytest.raises(ValueError, match="causally scoreable"):
        heldout_response_perplexity(
            FixedLogitPolicy(logits),
            input_ids,
            attention_mask,
            t.tensor([[1, IGNORE_INDEX, IGNORE_INDEX]], dtype=t.int64),
        )
    with pytest.raises(ValueError, match="loss token"):
        heldout_response_perplexity(
            FixedLogitPolicy(logits),
            input_ids,
            attention_mask,
            t.full_like(labels, IGNORE_INDEX),
        )


def test_perplexity_rejects_malformed_policy_outputs() -> None:
    input_ids = t.tensor([[1, 2, 3]], dtype=t.int64)
    attention_mask = t.ones_like(input_ids, dtype=t.bool)
    labels = t.tensor([[IGNORE_INDEX, 2, 3]], dtype=t.int64)
    with pytest.raises(TypeError, match="rank-3"):
        heldout_response_perplexity(
            FixedLogitPolicy([1, 2, 3]),
            input_ids,
            attention_mask,
            labels,
        )
    with pytest.raises(ValueError, match="match"):
        heldout_response_perplexity(
            FixedLogitPolicy(t.zeros((1, 2, 4))),
            input_ids,
            attention_mask,
            labels,
        )
    nonfinite = t.zeros((1, 3, 4))
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        heldout_response_perplexity(
            FixedLogitPolicy(nonfinite),
            input_ids,
            attention_mask,
            labels,
        )
