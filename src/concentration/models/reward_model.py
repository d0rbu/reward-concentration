"""Sequence-classification reward models and verified raw-score caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch as t
from beartype import beartype
from jaxtyping import Float32, jaxtyped
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from concentration.config import ModelDType, RewardModelConfig
from concentration.run_logging import RunLogger
from concentration.types import ScoreBatch

CACHE_VERSION = 1
SCORES_FILENAME = "scores.safetensors"
INDEX_FILENAME = "index.json"
RawScores = Float32[t.Tensor, "batch"]


class RewardModel(Protocol):
    """Swappable frozen raw-score reward-model boundary."""

    @property
    def rm_id(self) -> str: ...  # pragma: no cover - protocol declaration

    @jaxtyped(typechecker=beartype)
    def score(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> RawScores: ...  # pragma: no cover - protocol declaration


def _torch_dtype(dtype: ModelDType) -> t.dtype:
    return {ModelDType.FLOAT32: t.float32, ModelDType.BFLOAT16: t.bfloat16}[dtype]


@beartype
def chat_template_hash(tokenizer: Any) -> str:
    """Hash the reward model's exact tokenizer chat template."""
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise ValueError("reward-model tokenizer must expose one non-empty chat template")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


@beartype
def render_reward_chat(tokenizer: Any, prompt: str, response: str) -> str:
    """Render one prompt-response pair with the reward model's own non-thinking template."""
    if not prompt.strip() or not response.strip():
        raise ValueError("reward-model prompts and responses must be non-blank")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise TypeError("reward-model chat template must render a non-empty string")
    return rendered


class SequenceClassificationRewardModel:
    """Frozen Hugging Face sequence classifier returning unmodified scalar logits."""

    @beartype
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        rm_id: str,
        batch_size: int,
        max_length: int,
        device: str,
    ) -> None:
        if not rm_id.strip():
            raise ValueError("rm_id must be non-empty")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be positive")
        if not device:
            raise ValueError("device must be non-empty")
        self._model = model.to(device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._tokenizer = tokenizer
        self._rm_id = rm_id
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device
        self._chat_template_hash = chat_template_hash(tokenizer)

    @classmethod
    @beartype
    def from_config(cls, config: RewardModelConfig) -> SequenceClassificationRewardModel:
        """Load the configured tokenizer and scalar sequence classifier."""
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id,
            revision=config.revision,
            dtype=_torch_dtype(config.dtype),
        )
        return cls(
            model,
            tokenizer,
            rm_id=config.model_id,
            batch_size=int(config.batch_size),
            max_length=int(config.max_length),
            device=config.device,
        )

    @property
    def rm_id(self) -> str:
        return self._rm_id

    @property
    def template_hash(self) -> str:
        return self._chat_template_hash

    @jaxtyped(typechecker=beartype)
    def score(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> RawScores:
        """Return raw classifier logits without centering, scaling, or calibration."""
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have equal lengths")
        if not prompts:
            raise ValueError("reward scoring requires a non-empty batch")
        all_scores: list[t.Tensor] = []
        for start in range(0, len(prompts), self._batch_size):
            rendered = [
                render_reward_chat(self._tokenizer, prompt, response)
                for prompt, response in zip(
                    prompts[start : start + self._batch_size],
                    responses[start : start + self._batch_size],
                    strict=True,
                )
            ]
            encoded = self._tokenizer(
                rendered,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            if not isinstance(input_ids, t.Tensor) or input_ids.ndim != 2:
                raise TypeError("reward tokenizer must return rank-2 torch input_ids")
            if input_ids.shape[1] > self._max_length:
                raise ValueError(
                    f"reward-model input length {input_ids.shape[1]} exceeds "
                    f"max_length {self._max_length}"
                )
            device_batch = {
                name: value.to(self._device) if isinstance(value, t.Tensor) else value
                for name, value in encoded.items()
            }
            with t.inference_mode():
                logits = self._model(**device_batch).logits
            if not isinstance(logits, t.Tensor) or logits.ndim != 2 or logits.shape[1] != 1:
                raise ValueError("reward model must emit logits with shape [batch, 1]")
            all_scores.append(logits[:, 0].float())
        scores = t.cat(all_scores, dim=0)
        if scores.shape[0] != len(prompts):
            raise RuntimeError("reward model returned the wrong number of scores")
        return ScoreBatch.from_tensor(scores).tensor


@beartype
def score_cache_key(
    rm_id: str,
    template_hash: str,
    prompt: str,
    response: str,
) -> str:
    """Compute sha256(rm_id || chat-template-hash || prompt || response)."""
    return hashlib.sha256(
        (rm_id + template_hash + prompt + response).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ScoreCache:
    """A verified immutable in-memory view of a safetensors/JSON score cache."""

    rm_id: str
    template_hash: str
    index: dict[str, int]
    scores: ScoreBatch

    @classmethod
    @beartype
    def build(
        cls,
        path: Path,
        *,
        rm_id: str,
        template_hash: str,
        prompts: list[str],
        responses: list[str],
        scores: ScoreBatch,
    ) -> ScoreCache:
        """Create a new cache directory and immediately reload/verify it."""
        if len(prompts) != len(responses) or len(prompts) != scores.tensor.shape[0]:
            raise ValueError("cache prompts, responses, and scores must have equal lengths")
        keys = [
            score_cache_key(rm_id, template_hash, prompt, response)
            for prompt, response in zip(prompts, responses, strict=True)
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("score cache inputs must contain unique prompt-response keys")
        path.mkdir(parents=True, exist_ok=False)
        save_file({"scores": scores.tensor.detach().cpu().contiguous()}, path / SCORES_FILENAME)
        payload = {
            "version": CACHE_VERSION,
            "rm_id": rm_id,
            "chat_template_hash": template_hash,
            "index": {key: index for index, key in enumerate(keys)},
        }
        (path / INDEX_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        cache = cls.load(path)
        cache.verify(rm_id, template_hash, prompts, responses)
        return cache

    @classmethod
    @beartype
    def load(cls, path: Path) -> ScoreCache:
        """Load a cache and validate its file, metadata, tensor, and index invariants."""
        tensors = load_file(path / SCORES_FILENAME, device="cpu")
        if set(tensors) != {"scores"}:
            raise ValueError("score safetensors file must contain exactly the 'scores' tensor")
        scores = ScoreBatch.from_tensor(tensors["scores"])
        payload = json.loads((path / INDEX_FILENAME).read_text(encoding="utf-8"))
        if set(payload) != {"version", "rm_id", "chat_template_hash", "index"}:
            raise ValueError("score-cache index JSON has unexpected fields")
        if payload["version"] != CACHE_VERSION:
            raise ValueError(f"unsupported score-cache version {payload['version']}")
        rm_id = payload["rm_id"]
        template_hash = payload["chat_template_hash"]
        index = payload["index"]
        if not isinstance(rm_id, str) or not rm_id:
            raise ValueError("score-cache rm_id must be non-empty")
        if not isinstance(template_hash, str) or len(template_hash) != 64:
            raise ValueError("score-cache chat-template hash must be a SHA-256 hex digest")
        if not isinstance(index, dict) or not all(
            isinstance(key, str) and len(key) == 64 and type(value) is int
            for key, value in index.items()
        ):
            raise ValueError("score-cache index must map SHA-256 keys to integer offsets")
        expected_indices = set(range(scores.tensor.shape[0]))
        if set(index.values()) != expected_indices:
            raise ValueError("score-cache index must cover every tensor row exactly once")
        return cls(rm_id=rm_id, template_hash=template_hash, index=index, scores=scores)

    @beartype
    def verify(
        self,
        rm_id: str,
        template_hash: str,
        prompts: list[str],
        responses: list[str],
    ) -> None:
        """Assert metadata identity and exact cache completeness for a fixed dataset."""
        if self.rm_id != rm_id:
            raise ValueError(f"score-cache rm_id mismatch: {self.rm_id!r} != {rm_id!r}")
        if self.template_hash != template_hash:
            raise ValueError("score-cache chat-template hash mismatch")
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have equal lengths")
        expected = [
            score_cache_key(rm_id, template_hash, prompt, response)
            for prompt, response in zip(prompts, responses, strict=True)
        ]
        if len(set(expected)) != len(expected):
            raise ValueError("cache verification inputs must contain unique keys")
        if set(expected) != set(self.index):
            missing = set(expected) - set(self.index)
            extra = set(self.index) - set(expected)
            raise KeyError(f"score cache is incomplete: missing={len(missing)} extra={len(extra)}")

    @beartype
    def lookup(self, prompts: list[str], responses: list[str]) -> ScoreBatch:
        """Look up raw scores in request order; any missing key crashes."""
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have equal lengths")
        if not prompts:
            raise ValueError("score-cache lookup requires a non-empty batch")
        indices: list[int] = []
        for prompt, response in zip(prompts, responses, strict=True):
            key = score_cache_key(self.rm_id, self.template_hash, prompt, response)
            if key not in self.index:
                raise KeyError(f"missing reward score for key {key}")
            indices.append(self.index[key])
        index_tensor = t.tensor(indices, dtype=t.int64, device=self.scores.tensor.device)
        return ScoreBatch.from_tensor(self.scores.tensor[index_tensor])


@dataclass(frozen=True, slots=True)
class ScoreDiagnostics:
    """Logged-only raw-score moments; these values never transform score targets."""

    mean: float
    standard_deviation: float
    count: int


@beartype
def compute_score_diagnostics(scores: ScoreBatch) -> ScoreDiagnostics:
    """Compute population mean/std without modifying or replacing raw scores."""
    return ScoreDiagnostics(
        mean=float(scores.tensor.mean()),
        standard_deviation=float(scores.tensor.std(unbiased=False)),
        count=scores.tensor.shape[0],
    )


@beartype
def log_score_diagnostics(
    logger: RunLogger,
    scores: ScoreBatch,
    *,
    step: int,
    split: str = "train",
) -> ScoreBatch:
    """Log raw-score moments and return the identical untransformed score wrapper."""
    if not split.strip():
        raise ValueError("diagnostic split name must be non-empty")
    diagnostics = compute_score_diagnostics(scores)
    logger.log(
        step,
        {
            f"{split}/rm_score_mean": diagnostics.mean,
            f"{split}/rm_score_std": diagnostics.standard_deviation,
            f"{split}/rm_score_count": diagnostics.count,
        },
    )
    return scores
