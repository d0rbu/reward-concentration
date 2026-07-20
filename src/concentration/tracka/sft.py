"""Safety-SFT data derivation and TRL training over offset-derived chat spans."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
import trl
from beartype import beartype
from datasets import Dataset
from jaxtyping import Bool, Int, Int64, jaxtyped
from trl import SFTConfig, SFTTrainer

from concentration.config import ModelDType, SFTTrainConfig, WandbMode
from concentration.data.preference import (
    PreferenceResponse,
    TokenizedConversation,
    TokenizedPreferenceSplits,
    TokenSpan,
    build_prompt_splits,
    load_preference_dataset,
    tokenize_preference_splits,
)
from concentration.models.policy import load_policy
from concentration.seeding import seed_all
from concentration.types import IGNORE_INDEX

BatchTokenIds = Int64[t.Tensor, "batch tokens"]
BatchTokenMask = Bool[t.Tensor, "batch tokens"] | Int[t.Tensor, "batch tokens"]
TokenLabels = Int64[t.Tensor, "tokens"]
EFFECTIVE_CONFIG_FILENAME = "effective-config.json"
RUN_MANIFEST_FILENAME = "run-manifest.json"
TRAIN_LOGS_FILENAME = "train-logs.jsonl"


@beartype
def require_pad_token_id(value: object) -> int:
    """Refine a tokenizer's pad token id; padding with a fake or missing id corrupts batches."""
    if type(value) is not int or value < 0:
        raise ValueError("pad_token_id must be a non-negative integer")
    return value


@beartype
def warmup_steps_for(config: SFTTrainConfig) -> int:
    """Integer warmup steps: ceil(max_steps x warmup_frac).

    Passing the fraction itself to transformers is version-dependent (floats >= 1 are
    absolute steps, and 4.x ignores fractions entirely), so the mapping is explicit.
    """
    return math.ceil(int(config.max_steps) * float(config.warmup_frac))


@dataclass(frozen=True, slots=True)
class SFTItem:
    """One unique train-split response that is preferred in at least one kept pair."""

    source: PreferenceResponse
    conversation: TokenizedConversation
    labels: TokenLabels


@dataclass(frozen=True, slots=True)
class SFTDatasetCounts:
    """Auditable counts for preferred-response selection from the tokenized train split."""

    train_pairs: int
    train_unique_responses: int
    train_overlong_responses_removed: int
    train_incomplete_pairs_removed: int
    sft_items: int
    duplicate_preferred_selections_removed: int
    dual_labeled_sft_items: int


@dataclass(frozen=True, slots=True)
class SFTDataset:
    """Trainer-ready preferred-response items and their derivation counts."""

    items: tuple[SFTItem, ...]
    counts: SFTDatasetCounts

    def to_huggingface(self) -> Dataset:
        """Materialize only the two pretokenized columns consumed by TRL."""
        return Dataset.from_dict(
            {
                "input_ids": [item.conversation.input_ids.tolist() for item in self.items],
                "labels": [item.labels.tolist() for item in self.items],
            }
        )


@dataclass(frozen=True, slots=True)
class SFTResult:
    """Final training facts returned to tests and command callers."""

    output_dir: Path
    dataset_counts: SFTDatasetCounts
    final_metrics: dict[str, int | float]
    loss_history: tuple[float, ...]


@jaxtyped(typechecker=beartype)
def mask_loss_spans(
    input_ids: BatchTokenIds,
    attention_mask: BatchTokenMask,
    loss_spans: tuple[TokenSpan, ...],
) -> BatchTokenIds:
    """Keep token IDs inside each loss span and mask prompt and padding positions."""
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain a non-empty batch and sequence")
    if len(loss_spans) != input_ids.shape[0]:
        raise ValueError("loss_spans must contain exactly one span per batch item")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("attention_mask values must be binary")
    valid = attention_mask.bool()
    positions = t.arange(input_ids.shape[1], device=input_ids.device)
    selected_rows: list[t.Tensor] = []
    for row_index, span in enumerate(loss_spans):
        if span.stop > input_ids.shape[1]:
            raise ValueError("loss span exceeds the padded sequence length")
        row_valid = valid[row_index]
        valid_positions = positions[row_valid]
        if valid_positions.numel() == 0:
            raise ValueError("every batch item must contain at least one non-padding token")
        if int(valid_positions[-1] - valid_positions[0] + 1) != int(valid_positions.numel()):
            raise ValueError("non-padding tokens must form one contiguous span")
        if not bool(row_valid[span.start : span.stop].all()):
            raise ValueError("loss spans must lie entirely inside non-padding tokens")
        selected_rows.append(
            row_valid & positions.ge(span.start) & positions.lt(span.stop)
        )
    selected = t.stack(selected_rows)
    return input_ids.masked_fill(~selected, IGNORE_INDEX)


@beartype
def labels_for_conversation(conversation: TokenizedConversation) -> TokenLabels:
    """Construct exact loss-span labels for one unbatched conversation."""
    return mask_loss_spans(
        conversation.input_ids.unsqueeze(0),
        conversation.attention_mask.unsqueeze(0),
        (conversation.loss_span,),
    )[0]


@beartype
def build_sft_dataset(splits: TokenizedPreferenceSplits) -> SFTDataset:
    """Select train-only preferred responses, deduplicated by ``(prompt, response)``.

    Selection is existential: a response preferred in any kept pair is an SFT item even if
    another pair labels that same key as dispreferred. The source corpus has 579 such
    dual-labeled keys. Resolving labels for later reward-direction fitting is deliberately
    deferred; SFT only asks whether a response was ever selected as the better response.
    """
    train = splits.train
    train_prompts = {response.source.prompt for response in train.responses}
    heldout_prompts = {
        response.source.prompt
        for split in (splits.heldout_probe_train, splits.heldout_probe_test)
        for response in split.responses
    }
    if train_prompts & heldout_prompts:
        raise ValueError("SFT train prompts must be disjoint from both held-out splits")
    tokenized_by_key = {response.source.key: response for response in train.responses}
    if len(tokenized_by_key) != len(train.responses):
        raise ValueError("tokenized train responses must be unique by (prompt, response)")

    preferred_keys: list[tuple[str, str]] = []
    preferred_key_set: set[tuple[str, str]] = set()
    dispreferred_keys: set[tuple[str, str]] = set()
    for pair in train.pairs:
        if pair.better_response_id == 0:
            preferred_key = (pair.prompt, pair.response_0)
            dispreferred_key = (pair.prompt, pair.response_1)
        elif pair.better_response_id == 1:
            preferred_key = (pair.prompt, pair.response_1)
            dispreferred_key = (pair.prompt, pair.response_0)
        else:
            raise ValueError("better_response_id must be 0 or 1")
        if preferred_key not in tokenized_by_key or dispreferred_key not in tokenized_by_key:
            raise ValueError("every kept pair response must exist in the tokenized train responses")
        dispreferred_keys.add(dispreferred_key)
        if preferred_key not in preferred_key_set:
            preferred_keys.append(preferred_key)
            preferred_key_set.add(preferred_key)

    if not preferred_keys:
        raise ValueError("SFT selection produced an empty dataset")
    items = tuple(
        SFTItem(
            source=tokenized_by_key[key].source,
            conversation=tokenized_by_key[key].conversation,
            labels=labels_for_conversation(tokenized_by_key[key].conversation),
        )
        for key in preferred_keys
    )
    counts = SFTDatasetCounts(
        train_pairs=len(train.pairs),
        train_unique_responses=len(train.responses),
        train_overlong_responses_removed=train.overlong_responses_removed,
        train_incomplete_pairs_removed=train.incomplete_pairs_removed,
        sft_items=len(items),
        duplicate_preferred_selections_removed=len(train.pairs) - len(items),
        dual_labeled_sft_items=len(preferred_key_set & dispreferred_keys),
    )
    return SFTDataset(items=items, counts=counts)


def _int64_vector(value: object, name: str) -> t.Tensor:
    tensor = t.as_tensor(value)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty rank-1 sequence")
    if tensor.dtype is not t.int64:
        raise TypeError(f"{name} must contain int64 values")
    return tensor


@dataclass(frozen=True, slots=True)
class SFTDataCollator:
    """Right-pad pretokenized examples without changing their offset-derived labels."""

    pad_token_id: int

    def __post_init__(self) -> None:
        require_pad_token_id(self.pad_token_id)

    @beartype
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, t.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty feature list")
        if any(set(feature) != {"input_ids", "labels"} for feature in features):
            raise ValueError("SFT features must contain exactly input_ids and labels")
        input_ids = [_int64_vector(feature["input_ids"], "input_ids") for feature in features]
        labels = [_int64_vector(feature["labels"], "labels") for feature in features]
        if any(ids.shape != item_labels.shape for ids, item_labels in zip(input_ids, labels, strict=True)):
            raise ValueError("each labels sequence must match its input_ids sequence")
        if any(
            bool((item_labels.ne(IGNORE_INDEX) & item_labels.ne(ids)).any())
            for ids, item_labels in zip(input_ids, labels, strict=True)
        ):
            raise ValueError("unmasked labels must equal input_ids")
        if any(not bool(item_labels.ne(IGNORE_INDEX).any()) for item_labels in labels):
            raise ValueError("every SFT item must contain at least one loss token")
        if any(not bool(item_labels[1:].ne(IGNORE_INDEX).any()) for item_labels in labels):
            raise ValueError("every SFT item must contain a causally scoreable loss token")
        lengths = t.tensor([ids.shape[0] for ids in input_ids], dtype=t.int64)
        padded_ids = t.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        padded_labels = t.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        positions = t.arange(padded_ids.shape[1]).expand(padded_ids.shape[0], -1)
        return {
            "input_ids": padded_ids,
            "attention_mask": positions < lengths[:, None],
            "labels": padded_labels,
        }


def _require_available_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"SFT output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty SFT output directory: {output_dir}")


@beartype
def effective_config(config: SFTTrainConfig) -> dict[str, Any]:
    """Return the fully defaulted, JSON-serializable SFT configuration."""
    return asdict(config)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _configure_wandb(config: SFTTrainConfig) -> list[str]:
    os.environ["WANDB_MODE"] = config.wandb.mode.value
    os.environ["WANDB_PROJECT"] = config.wandb.project
    return [] if config.wandb.mode is WandbMode.DISABLED else ["wandb"]


def _train_sft(
    config: SFTTrainConfig,
    model: t.nn.Module,
    tokenizer: Any,
    splits: TokenizedPreferenceSplits,
) -> SFTResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / EFFECTIVE_CONFIG_FILENAME, effective_config(config))
    pad_token_id = require_pad_token_id(tokenizer.pad_token_id)

    selected = build_sft_dataset(splits)
    trainer_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.per_device_batch_size),
        gradient_accumulation_steps=int(config.gradient_accumulation_steps),
        max_steps=int(config.max_steps),
        learning_rate=float(config.learning_rate),
        warmup_steps=warmup_steps_for(config),
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_strategy="no",
        report_to=_configure_wandb(config),
        run_name=output_dir.name,
        seed=int(config.seed),
        data_seed=int(config.seed),
        full_determinism=True,
        use_cpu=config.policy.device == "cpu",
        bf16=config.policy.dtype is ModelDType.BFLOAT16,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        packing=False,
        completion_only_loss=False,
        loss_type="nll",
        do_train=True,
        disable_tqdm=True,
    )
    trainer = SFTTrainer(
        model=cast(Any, model),
        args=trainer_config,
        data_collator=SFTDataCollator(pad_token_id),
        train_dataset=selected.to_huggingface(),
        processing_class=tokenizer,
    )
    train_output = trainer.train()
    final_metrics = dict(train_output.metrics)
    loss_history = tuple(
        float(record["loss"])
        for record in trainer.state.log_history
        if "loss" in record
    )
    if not loss_history or not all(math.isfinite(loss) for loss in loss_history):
        raise FloatingPointError("SFT loss history must be non-empty and finite")
    train_loss = final_metrics.get("train_loss")
    if not isinstance(train_loss, int | float) or not math.isfinite(train_loss):
        raise FloatingPointError("SFT final train_loss must be finite")

    with (output_dir / TRAIN_LOGS_FILENAME).open("x", encoding="utf-8") as stream:
        for record in trainer.state.log_history:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    saved_model = trainer.accelerator.unwrap_model(trainer.model)
    saved_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    manifest = {
        "schema_version": 1,
        "trl_version": trl.__version__,
        "config": effective_config(config),
        "dataset_counts": asdict(selected.counts),
        "final_metrics": final_metrics,
        "loss_history": loss_history,
    }
    _write_json(output_dir / RUN_MANIFEST_FILENAME, manifest)
    return SFTResult(
        output_dir=output_dir,
        dataset_counts=selected.counts,
        final_metrics=final_metrics,
        loss_history=loss_history,
    )


@beartype
def train_sft(
    config: SFTTrainConfig,
    model: t.nn.Module,
    tokenizer: Any,
    splits: TokenizedPreferenceSplits,
) -> SFTResult:
    """Train from injected pipeline components, primarily for deterministic integration tests."""
    output_dir = Path(config.output_dir)
    _require_available_output_dir(output_dir)
    seed_all(config.seed)
    return _train_sft(config, model, tokenizer, splits)


@beartype
def run_sft(config: SFTTrainConfig) -> SFTResult:
    """Load the configured policy/data pipeline and produce a safety-SFT checkpoint."""
    output_dir = Path(config.output_dir)
    _require_available_output_dir(output_dir)
    seed_all(config.seed)
    loaded = load_policy(config.policy)
    raw_dataset = load_preference_dataset(config.data.preference_dataset_id)
    splits = build_prompt_splits(raw_dataset, config.data)
    tokenized = tokenize_preference_splits(splits, loaded.tokenizer, config.data)
    return _train_sft(config, loaded.model, loaded.tokenizer, tokenized)
