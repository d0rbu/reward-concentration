"""Command-line entry points for safety-SFT and held-out response perplexity."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
from beartype import beartype

from concentration.config import (
    DataConfig,
    ModelConfig,
    SFTTrainConfig,
    WandbConfig,
)
from concentration.data.preference import (
    build_prompt_splits,
    load_preference_dataset,
    tokenize_preference_splits,
)
from concentration.eval.capability import heldout_response_perplexity
from concentration.models.policy import load_policy
from concentration.seeding import seed_all
from concentration.tracka.sft import (
    SFTDataCollator,
    labels_for_conversation,
    require_pad_token_id,
    run_sft,
)
from concentration.types import Rank, parse_rank

SECTION_KEYS = {
    "policy": frozenset({"model_id", "revision", "dtype", "device"}),
    "data": frozenset(
        {
            "preference_dataset_id",
            "safety_eval_dataset_id",
            "max_len",
            "heldout_probe_train_frac",
            "heldout_probe_test_frac",
            "seed",
        }
    ),
    "sft": frozenset(
        {
            "seed",
            "learning_rate",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "warmup_frac",
            "output_dir",
        }
    ),
    "wandb": frozenset({"mode", "project"}),
}
TOP_LEVEL_KEYS = frozenset(SECTION_KEYS)


@dataclass(frozen=True, slots=True)
class PPLRunResult:
    """Aggregated held-out response perplexity command result."""

    checkpoint: str
    evaluated_responses: int
    token_count: int
    mean_nll: float
    perplexity: float


def _mapping_section(raw: Mapping[str, object], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"TOML section [{name}] must be a table")
    unknown = set(value) - SECTION_KEYS[name]
    if unknown:
        raise ValueError(f"unknown keys in [{name}]: {sorted(unknown)}")
    return cast(dict[str, Any], value)


@beartype
def parse_sft_config(raw: Mapping[str, object]) -> SFTTrainConfig:
    """Parse one schema-locked TOML mapping through every config ``from_raw`` boundary."""
    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    policy = ModelConfig.from_raw(**_mapping_section(raw, "policy"))
    data = DataConfig.from_raw(**_mapping_section(raw, "data"))
    wandb = WandbConfig.from_raw(**_mapping_section(raw, "wandb"))
    return SFTTrainConfig.from_raw(
        policy=policy,
        data=data,
        wandb=wandb,
        **_mapping_section(raw, "sft"),
    )


@beartype
def load_sft_config(path: Path) -> SFTTrainConfig:
    """Load a TOML file and return its fully refined SFT configuration."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return parse_sft_config(raw)


@beartype
def run_ppl(
    config: SFTTrainConfig,
    checkpoint: str,
    count: Rank | None,
    batch_size: Rank,
) -> PPLRunResult:
    """Evaluate a checkpoint on Phase 1 ``heldout_probe_train`` responses."""
    if not checkpoint.strip():
        raise ValueError("checkpoint must be non-empty")
    seed_all(config.seed)
    checkpoint_config = ModelConfig.from_raw(
        model_id=checkpoint,
        revision=config.policy.revision,
        dtype=config.policy.dtype.value,
        device=config.policy.device,
    )
    loaded = load_policy(checkpoint_config)
    raw_dataset = load_preference_dataset(config.data.preference_dataset_id)
    splits = build_prompt_splits(raw_dataset, config.data)
    tokenized = tokenize_preference_splits(splits, loaded.tokenizer, config.data)
    heldout = tokenized.heldout_probe_train.responses
    selected = heldout if count is None else heldout[: int(count)]
    collator = SFTDataCollator(require_pad_token_id(loaded.tokenizer.pad_token_id))
    nll_sum = t.zeros((), dtype=t.float32)
    token_count = 0
    for start in range(0, len(selected), int(batch_size)):
        response_batch = selected[start : start + int(batch_size)]
        tensors = collator(
            [
                {
                    "input_ids": response.conversation.input_ids,
                    "labels": labels_for_conversation(response.conversation),
                }
                for response in response_batch
            ]
        )
        metrics = heldout_response_perplexity(
            loaded.model,
            tensors["input_ids"].to(config.policy.device),
            tensors["attention_mask"].to(config.policy.device),
            tensors["labels"].to(config.policy.device),
        )
        nll_sum += metrics.nll_sum.cpu()
        token_count += metrics.token_count
    mean_nll = nll_sum / token_count
    perplexity = mean_nll.exp()
    return PPLRunResult(
        checkpoint=checkpoint,
        evaluated_responses=len(selected),
        token_count=token_count,
        mean_nll=float(mean_nll),
        perplexity=float(perplexity),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="concentration")
    subcommands = parser.add_subparsers(dest="command", required=True)
    sft_parser = subcommands.add_parser("sft", help="train the safety-SFT checkpoint")
    sft_parser.add_argument("config", type=Path, help="TOML SFT configuration")
    ppl_parser = subcommands.add_parser(
        "ppl",
        help="evaluate heldout_probe_train response perplexity",
    )
    ppl_parser.add_argument("config", type=Path, help="TOML SFT/data configuration")
    ppl_parser.add_argument("checkpoint", help="local or Hugging Face checkpoint")
    ppl_parser.add_argument("--count", type=int, default=None, help="maximum response count")
    ppl_parser.add_argument("--batch-size", type=int, default=8)
    return parser


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to a runnable workflow function."""
    args = _parser().parse_args(argv)
    config = load_sft_config(args.config)
    if args.command == "sft":
        run_sft(config)
    elif args.command == "ppl":
        count = None if args.count is None else parse_rank(args.count)
        result = run_ppl(config, args.checkpoint, count, parse_rank(args.batch_size))
        print(json.dumps(asdict(result), sort_keys=True, allow_nan=False))
    return 0
