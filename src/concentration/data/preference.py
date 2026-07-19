"""PKU single-dimension preference loading, splitting, and chat tokenization."""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from typing import Any

import torch as t
from datasets import DatasetDict, Features, Value, load_dataset
from jaxtyping import Bool, Int64

from concentration.config import PREFERENCE_DATASET_ID, DataConfig

LOGGER = logging.getLogger(__name__)

EXPECTED_PREFERENCE_FEATURES = Features(
    {
        "prompt": Value("string"),
        "response_0": Value("string"),
        "response_1": Value("string"),
        "prompt_source": Value("string"),
        "response_0_source": Value("string"),
        "response_1_source": Value("string"),
        "better_response_id": Value("int64"),
        "response_0_sha256": Value("string"),
        "response_1_sha256": Value("string"),
    }
)
EXPECTED_PREFERENCE_SPLITS = frozenset({"train", "test"})
OBSERVED_PREFERENCE_ROWS = {"train": 72_996, "test": 8_109}
SPLIT_NAMES = ("train", "heldout_probe_train", "heldout_probe_test")
TokenIds = Int64[t.Tensor, "tokens"]
TokenMask = Bool[t.Tensor, "tokens"]


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """A non-empty half-open token interval."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.stop) is not int:
            raise TypeError("token-span boundaries must be integers")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("token spans must be non-empty and ordered")


@dataclass(frozen=True, slots=True)
class TokenizedConversation:
    """One rendered prompt-response conversation and its response spans."""

    input_ids: TokenIds
    attention_mask: TokenMask
    loss_span: TokenSpan
    pool_span: TokenSpan


@dataclass(frozen=True, slots=True)
class PreferencePair:
    """One source comparison, retaining its pair-relative preference label."""

    prompt: str
    response_0: str
    response_1: str
    response_0_sha256: str
    response_1_sha256: str
    better_response_id: int


@dataclass(frozen=True, slots=True)
class PreferenceResponse:
    """One unique non-blank (prompt, response) training item."""

    prompt: str
    response: str
    response_sha256: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.prompt, self.response)


@dataclass(frozen=True, slots=True)
class PreferenceSplit:
    """Pair comparisons and deduplicated responses for one prompt-disjoint split."""

    pairs: tuple[PreferencePair, ...]
    responses: tuple[PreferenceResponse, ...]
    duplicate_responses_removed: int
    blank_responses_removed: int
    pairs_with_blank_response: int


@dataclass(frozen=True, slots=True)
class PreferenceSplits:
    """The three required prompt-level splits."""

    train: PreferenceSplit
    heldout_probe_train: PreferenceSplit
    heldout_probe_test: PreferenceSplit

    def items(self) -> tuple[tuple[str, PreferenceSplit], ...]:
        return (
            ("train", self.train),
            ("heldout_probe_train", self.heldout_probe_train),
            ("heldout_probe_test", self.heldout_probe_test),
        )


@dataclass(frozen=True, slots=True)
class TokenizedPreferenceResponse:
    """A unique preference response after rendering and length validation."""

    source: PreferenceResponse
    conversation: TokenizedConversation


@dataclass(frozen=True, slots=True)
class TokenizedPreferenceSplit:
    """A tokenized split plus explicit filter counts."""

    pairs: tuple[PreferencePair, ...]
    responses: tuple[TokenizedPreferenceResponse, ...]
    overlong_responses_removed: int
    incomplete_pairs_removed: int


@dataclass(frozen=True, slots=True)
class TokenizedPreferenceSplits:
    """Tokenized forms of all prompt-level preference splits."""

    train: TokenizedPreferenceSplit
    heldout_probe_train: TokenizedPreferenceSplit
    heldout_probe_test: TokenizedPreferenceSplit

    def items(self) -> tuple[tuple[str, TokenizedPreferenceSplit], ...]:
        return (
            ("train", self.train),
            ("heldout_probe_train", self.heldout_probe_train),
            ("heldout_probe_test", self.heldout_probe_test),
        )


def assert_preference_schema(dataset: DatasetDict) -> None:
    """Crash unless both upstream splits exactly match the observed feature schema."""
    if frozenset(dataset.keys()) != EXPECTED_PREFERENCE_SPLITS:
        raise ValueError(
            f"preference splits changed: expected {sorted(EXPECTED_PREFERENCE_SPLITS)}, "
            f"got {sorted(dataset.keys())}"
        )
    for split_name in sorted(EXPECTED_PREFERENCE_SPLITS):
        split = dataset[split_name]
        if split.features != EXPECTED_PREFERENCE_FEATURES:
            raise ValueError(
                f"preference schema changed for {split_name}: "
                f"expected {EXPECTED_PREFERENCE_FEATURES}, got {split.features}"
            )
        if split.column_names != list(EXPECTED_PREFERENCE_FEATURES):
            raise ValueError(f"preference column order changed for {split_name}")


def load_preference_dataset(dataset_id: str = PREFERENCE_DATASET_ID) -> DatasetDict:
    """Materialize the PKU preference dataset and assert its exact schema."""
    loaded = load_dataset(dataset_id)
    if not isinstance(loaded, DatasetDict):
        raise TypeError("preference loader must return a DatasetDict")
    assert_preference_schema(loaded)
    return loaded


def _source_sha256(prompt: str, response: str) -> str:
    return hashlib.sha256((prompt + response).encode("utf-8")).hexdigest()


def _parse_pair(row: dict[str, Any]) -> PreferencePair:
    prompt = row["prompt"]
    response_0 = row["response_0"]
    response_1 = row["response_1"]
    if not all(isinstance(value, str) for value in (prompt, response_0, response_1)):
        raise TypeError("preference text fields must be strings")
    if not prompt.strip():
        raise ValueError("preference prompts must be non-blank")
    better_response_id = int(row["better_response_id"])
    if better_response_id not in (0, 1):
        raise ValueError("better_response_id must be 0 or 1")
    response_0_sha256 = row["response_0_sha256"]
    response_1_sha256 = row["response_1_sha256"]
    if response_0_sha256 != _source_sha256(prompt, response_0):
        raise ValueError("response_0_sha256 does not match prompt + response_0")
    if response_1_sha256 != _source_sha256(prompt, response_1):
        raise ValueError("response_1_sha256 does not match prompt + response_1")
    return PreferencePair(
        prompt=prompt,
        response_0=response_0,
        response_1=response_1,
        response_0_sha256=response_0_sha256,
        response_1_sha256=response_1_sha256,
        better_response_id=better_response_id,
    )


def _deduplicate_responses(pairs: tuple[PreferencePair, ...]) -> PreferenceSplit:
    responses: dict[tuple[str, str], PreferenceResponse] = {}
    duplicates = 0
    blanks = 0
    pairs_with_blank = 0
    valid_pairs: list[PreferencePair] = []
    for pair in pairs:
        pair_has_blank = False
        for response, digest in (
            (pair.response_0, pair.response_0_sha256),
            (pair.response_1, pair.response_1_sha256),
        ):
            if not response.strip():
                blanks += 1
                pair_has_blank = True
                continue
            key = (pair.prompt, response)
            existing = responses.get(key)
            if existing is not None:
                if existing.response_sha256 != digest:
                    raise ValueError("duplicate response has inconsistent SHA-256")
                duplicates += 1
                continue
            responses[key] = PreferenceResponse(pair.prompt, response, digest)
        if pair_has_blank:
            pairs_with_blank += 1
        else:
            valid_pairs.append(pair)
    return PreferenceSplit(
        pairs=tuple(valid_pairs),
        responses=tuple(responses.values()),
        duplicate_responses_removed=duplicates,
        blank_responses_removed=blanks,
        pairs_with_blank_response=pairs_with_blank,
    )


def build_prompt_splits(dataset: DatasetDict, config: DataConfig) -> PreferenceSplits:
    """Merge upstream splits, then create deterministic prompt-disjoint project splits."""
    assert_preference_schema(dataset)
    pairs_by_prompt: dict[str, list[PreferencePair]] = {}
    for upstream_split in ("train", "test"):
        for row in dataset[upstream_split]:
            pair = _parse_pair(row)
            pairs_by_prompt.setdefault(pair.prompt, []).append(pair)

    prompts = sorted(pairs_by_prompt)
    random.Random(int(config.seed)).shuffle(prompts)
    probe_test_count = int(len(prompts) * config.heldout_probe_test_frac)
    probe_train_count = int(len(prompts) * config.heldout_probe_train_frac)
    train_count = len(prompts) - probe_train_count - probe_test_count
    if min(train_count, probe_train_count, probe_test_count) <= 0:
        raise ValueError("prompt split configuration produced an empty split")

    prompt_partitions = {
        "heldout_probe_test": prompts[:probe_test_count],
        "heldout_probe_train": prompts[
            probe_test_count : probe_test_count + probe_train_count
        ],
        "train": prompts[probe_test_count + probe_train_count :],
    }
    built: dict[str, PreferenceSplit] = {}
    for split_name in SPLIT_NAMES:
        pairs = tuple(
            pair
            for prompt in prompt_partitions[split_name]
            for pair in pairs_by_prompt[prompt]
        )
        split = _deduplicate_responses(pairs)
        if not split.responses or not split.pairs:
            raise ValueError(f"preference split {split_name} is empty after blank filtering")
        built[split_name] = split
        LOGGER.info(
            "preference split %s: pairs=%d responses=%d duplicates_removed=%d "
            "blank_responses_removed=%d pairs_with_blank=%d",
            split_name,
            len(split.pairs),
            len(split.responses),
            split.duplicate_responses_removed,
            split.blank_responses_removed,
            split.pairs_with_blank_response,
        )
    return PreferenceSplits(
        train=built["train"],
        heldout_probe_train=built["heldout_probe_train"],
        heldout_probe_test=built["heldout_probe_test"],
    )


def _render_chat(tokenizer: Any, messages: list[dict[str, str]], *, continue_final: bool) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=continue_final,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer chat template must render a string")
    return rendered


def tokenize_chat(tokenizer: Any, prompt: str, response: str) -> TokenizedConversation:
    """Render a non-thinking chat and derive content/loss spans from offsets and lengths."""
    if not prompt.strip():
        raise ValueError("prompt must be non-blank")
    if not response.strip():
        raise ValueError("response must be non-blank")
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    probe = "span-boundary-probe"
    probe_messages = [messages[0], {"role": "assistant", "content": probe}]
    continued_probe = _render_chat(tokenizer, probe_messages, continue_final=True)
    continued_content = _render_chat(tokenizer, messages, continue_final=True)
    full_text = _render_chat(tokenizer, messages, continue_final=False)
    if not continued_probe.endswith(probe):
        raise ValueError("chat template transformed assistant content")
    content_start = len(continued_probe) - len(probe)
    if continued_content[:content_start] != continued_probe[:content_start]:
        raise ValueError("chat template prefix depends on assistant content")
    if not continued_content.endswith(response):
        raise ValueError("chat template transformed assistant content")
    content_stop = len(continued_content)
    if not full_text.startswith(continued_content):
        raise ValueError("continued chat is not a prefix of the closed chat")

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids_raw = encoded["input_ids"]
    offsets_raw = encoded["offset_mapping"]
    if input_ids_raw and isinstance(input_ids_raw[0], list):
        raise ValueError("chat tokenizer unexpectedly returned a batch")
    input_ids = t.tensor(input_ids_raw, dtype=t.int64)
    offsets = [(int(start), int(stop)) for start, stop in offsets_raw]
    content_tokens = [
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > content_start and start < content_stop
    ]
    if not content_tokens:
        raise ValueError("assistant content produced no tokens")
    if content_tokens != list(range(content_tokens[0], content_tokens[-1] + 1)):
        raise ValueError("assistant content tokens must form one contiguous span")
    pool_span = TokenSpan(content_tokens[0], content_tokens[-1] + 1)
    loss_span = TokenSpan(pool_span.start, input_ids.shape[0])
    if loss_span.stop <= pool_span.stop:
        raise ValueError("chat template must append an end-of-turn token after assistant content")
    return TokenizedConversation(
        input_ids=input_ids,
        attention_mask=t.ones_like(input_ids, dtype=t.bool),
        loss_span=loss_span,
        pool_span=pool_span,
    )


def _tokenize_split(
    split_name: str,
    split: PreferenceSplit,
    tokenizer: Any,
    max_len: int,
) -> TokenizedPreferenceSplit:
    tokenized_by_key: dict[tuple[str, str], TokenizedPreferenceResponse] = {}
    overlong = 0
    for response in split.responses:
        conversation = tokenize_chat(tokenizer, response.prompt, response.response)
        if conversation.input_ids.shape[0] > max_len:
            overlong += 1
            continue
        tokenized_by_key[response.key] = TokenizedPreferenceResponse(response, conversation)
    kept_pairs = tuple(
        pair
        for pair in split.pairs
        if (pair.prompt, pair.response_0) in tokenized_by_key
        and (pair.prompt, pair.response_1) in tokenized_by_key
    )
    incomplete_pairs = len(split.pairs) - len(kept_pairs)
    if not tokenized_by_key or not kept_pairs:
        raise ValueError(f"preference split {split_name} is empty after length filtering")
    LOGGER.info(
        "preference token filter %s: kept_responses=%d overlong_removed=%d "
        "kept_pairs=%d incomplete_pairs_removed=%d max_len=%d",
        split_name,
        len(tokenized_by_key),
        overlong,
        len(kept_pairs),
        incomplete_pairs,
        max_len,
    )
    return TokenizedPreferenceSplit(
        pairs=kept_pairs,
        responses=tuple(tokenized_by_key.values()),
        overlong_responses_removed=overlong,
        incomplete_pairs_removed=incomplete_pairs,
    )


def tokenize_preference_splits(
    splits: PreferenceSplits,
    tokenizer: Any,
    config: DataConfig,
) -> TokenizedPreferenceSplits:
    """Chat-template and length-filter every split, crashing if any becomes empty."""
    built = {
        split_name: _tokenize_split(split_name, split, tokenizer, int(config.max_len))
        for split_name, split in splits.items()
    }
    return TokenizedPreferenceSplits(
        train=built["train"],
        heldout_probe_train=built["heldout_probe_train"],
        heldout_probe_test=built["heldout_probe_test"],
    )
