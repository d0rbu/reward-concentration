"""Frozen, validated configuration objects and raw-boundary parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from beartype import beartype

from concentration.types import (
    NonNegativeFloat,
    Rank,
    Seed,
    UnitInterval,
    parse_non_negative_float,
    parse_rank,
    parse_seed,
    parse_unit_interval,
)

DEV_POLICY_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MAIN_POLICY_MODEL_ID = "Qwen/Qwen3-1.7B-Base"
DEFAULT_REWARD_MODEL_ID = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
DEV_REWARD_MODEL_ID = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
PREFERENCE_DATASET_ID = "PKU-Alignment/PKU-SafeRLHF-single-dimension"
SAFETY_EVAL_DATASET_ID = "PKU-Alignment/BeaverTails-Evaluation"


class Pooling(StrEnum):
    """Supported response-token pooling reductions."""

    MEAN = "mean"
    LAST = "last"
    MAX = "max"
    MIN = "min"


class ModelDType(StrEnum):
    """Explicit floating-point model-loading dtypes."""

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


class Nonlinearity(StrEnum):
    """Supported head nonlinearities."""

    RELU = "relu"
    SIGMOID = "sigmoid"


class RewardHeadMode(StrEnum):
    """Reward-head architectures."""

    LINEAR_PROBE = "linear_probe"
    MLP = "mlp"


class RewardHeadInit(StrEnum):
    """Reward-direction initialization algorithms."""

    DIFFMEANS = "diffmeans"
    PROBE = "probe"


class LambdaScheduleShape(StrEnum):
    """Gradient-reversal schedule shapes."""

    DANN = "dann"
    LINEAR = "linear"
    CONSTANT = "constant"
    COSINE = "cosine"


class TrackAAlgorithm(StrEnum):
    """Permitted Track-A preference-optimization algorithms."""

    PPO = "ppo"
    GRPO = "grpo"


class WandbMode(StrEnum):
    """Weights & Biases execution modes."""

    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"


def _require_nonempty(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_phantom(value: object, expected: type[object], name: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be refined as {expected.__name__}")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Policy model-loading configuration."""

    model_id: str = DEV_POLICY_MODEL_ID
    revision: str = "main"
    dtype: ModelDType = ModelDType.BFLOAT16
    device: str = "cuda"

    def __post_init__(self) -> None:
        _require_nonempty(self.model_id, "model_id")
        _require_nonempty(self.revision, "revision")
        _require_nonempty(self.device, "device")
        if not isinstance(self.dtype, ModelDType):
            raise TypeError("dtype must be refined as ModelDType")

    @classmethod
    @beartype
    def from_raw(
        cls,
        *,
        model_id: str = DEV_POLICY_MODEL_ID,
        revision: str = "main",
        dtype: str = "bfloat16",
        device: str = "cuda",
    ) -> ModelConfig:
        return cls(model_id=model_id, revision=revision, dtype=ModelDType(dtype), device=device)


@dataclass(frozen=True, slots=True)
class RewardModelConfig:
    """Sequence-classification reward-model loading and scoring configuration."""

    model_id: str = DEFAULT_REWARD_MODEL_ID
    revision: str = "main"
    dtype: ModelDType = ModelDType.BFLOAT16
    device: str = "cuda"
    batch_size: Rank = field(default_factory=lambda: parse_rank(8))
    max_length: Rank = field(default_factory=lambda: parse_rank(2048))

    def __post_init__(self) -> None:
        _require_nonempty(self.model_id, "model_id")
        _require_nonempty(self.revision, "revision")
        _require_nonempty(self.device, "device")
        if not isinstance(self.dtype, ModelDType):
            raise TypeError("dtype must be refined as ModelDType")
        _require_phantom(self.batch_size, Rank, "batch_size")
        _require_phantom(self.max_length, Rank, "max_length")

    @classmethod
    @beartype
    def from_raw(
        cls,
        *,
        model_id: str = DEFAULT_REWARD_MODEL_ID,
        revision: str = "main",
        dtype: str = "bfloat16",
        device: str = "cuda",
        batch_size: int | str = 8,
        max_length: int | str = 2048,
    ) -> RewardModelConfig:
        return cls(
            model_id=model_id,
            revision=revision,
            dtype=ModelDType(dtype),
            device=device,
            batch_size=parse_rank(batch_size),
            max_length=parse_rank(max_length),
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Preference and safety-evaluation data configuration."""

    preference_dataset_id: str = PREFERENCE_DATASET_ID
    safety_eval_dataset_id: str = SAFETY_EVAL_DATASET_ID
    max_len: Rank = field(default_factory=lambda: parse_rank(1024))
    heldout_probe_train_frac: UnitInterval = field(
        default_factory=lambda: parse_unit_interval(0.1)
    )
    heldout_probe_test_frac: UnitInterval = field(
        default_factory=lambda: parse_unit_interval(0.1)
    )
    seed: Seed = field(default_factory=lambda: parse_seed(0))

    def __post_init__(self) -> None:
        _require_nonempty(self.preference_dataset_id, "preference_dataset_id")
        _require_nonempty(self.safety_eval_dataset_id, "safety_eval_dataset_id")
        _require_phantom(self.max_len, Rank, "max_len")
        _require_phantom(
            self.heldout_probe_train_frac,
            UnitInterval,
            "heldout_probe_train_frac",
        )
        _require_phantom(
            self.heldout_probe_test_frac,
            UnitInterval,
            "heldout_probe_test_frac",
        )
        _require_phantom(self.seed, Seed, "seed")
        if self.heldout_probe_train_frac + self.heldout_probe_test_frac >= 1.0:
            raise ValueError("held-out split fractions must sum to less than one")

    @classmethod
    @beartype
    def from_raw(
        cls,
        *,
        preference_dataset_id: str = PREFERENCE_DATASET_ID,
        safety_eval_dataset_id: str = SAFETY_EVAL_DATASET_ID,
        max_len: int | str = 1024,
        heldout_probe_train_frac: float | int | str = 0.1,
        heldout_probe_test_frac: float | int | str = 0.1,
        seed: int | str = 0,
    ) -> DataConfig:
        return cls(
            preference_dataset_id=preference_dataset_id,
            safety_eval_dataset_id=safety_eval_dataset_id,
            max_len=parse_rank(max_len),
            heldout_probe_train_frac=parse_unit_interval(heldout_probe_train_frac),
            heldout_probe_test_frac=parse_unit_interval(heldout_probe_test_frac),
            seed=parse_seed(seed),
        )


@dataclass(frozen=True, slots=True)
class RepExtractionConfig:
    """Layer and response-pooling selection."""

    layer: int
    pooling: Pooling = Pooling.MEAN

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if not isinstance(self.pooling, Pooling):
            raise TypeError("pooling must be refined as Pooling")

    @classmethod
    @beartype
    def from_raw(cls, *, layer: int | str, pooling: str = "mean") -> RepExtractionConfig:
        if isinstance(layer, bool):
            raise TypeError("layer must be an integer, not bool")
        return cls(layer=int(layer), pooling=Pooling(pooling))


@dataclass(frozen=True, slots=True)
class RewardHeadConfig:
    """Reward bottleneck and head configuration."""

    rank_n: Rank = field(default_factory=lambda: parse_rank(1))
    nonlinearity: Nonlinearity = Nonlinearity.RELU
    mode: RewardHeadMode = RewardHeadMode.LINEAR_PROBE
    init: RewardHeadInit = RewardHeadInit.DIFFMEANS
    trainable: bool = True

    def __post_init__(self) -> None:
        _require_phantom(self.rank_n, Rank, "rank_n")
        if not isinstance(self.nonlinearity, Nonlinearity):
            raise TypeError("nonlinearity must be refined as Nonlinearity")
        if not isinstance(self.mode, RewardHeadMode):
            raise TypeError("mode must be refined as RewardHeadMode")
        if not isinstance(self.init, RewardHeadInit):
            raise TypeError("init must be refined as RewardHeadInit")
        if type(self.trainable) is not bool:
            raise TypeError("trainable must be bool")

    @classmethod
    @beartype
    def from_raw(
        cls,
        *,
        rank_n: int | str = 1,
        nonlinearity: str = "relu",
        mode: str = "linear_probe",
        init: str = "diffmeans",
        trainable: bool = True,
    ) -> RewardHeadConfig:
        return cls(
            rank_n=parse_rank(rank_n),
            nonlinearity=Nonlinearity(nonlinearity),
            mode=RewardHeadMode(mode),
            init=RewardHeadInit(init),
            trainable=trainable,
        )


@dataclass(frozen=True, slots=True)
class AdvHeadConfig:
    """Adversarial regression-head configuration; no hidden layers means linear."""

    hidden_dims: tuple[Rank, ...] = ()
    nonlinearity: Nonlinearity = Nonlinearity.RELU

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_dims, tuple) or any(
            not isinstance(dimension, Rank) for dimension in self.hidden_dims
        ):
            raise TypeError("hidden_dims must contain refined Rank values")
        if not isinstance(self.nonlinearity, Nonlinearity):
            raise TypeError("nonlinearity must be refined as Nonlinearity")

    @classmethod
    @beartype
    def from_raw(
        cls,
        *,
        hidden_dims: tuple[int | str, ...] = (),
        nonlinearity: str = "relu",
    ) -> AdvHeadConfig:
        return cls(
            hidden_dims=tuple(parse_rank(dimension) for dimension in hidden_dims),
            nonlinearity=Nonlinearity(nonlinearity),
        )


@dataclass(frozen=True, slots=True)
class LambdaScheduleConfig:
    """Gradient-reversal lambda schedule configuration."""

    shape: LambdaScheduleShape = LambdaScheduleShape.DANN
    max: NonNegativeFloat = field(default_factory=lambda: parse_non_negative_float(1.0))
    warmup_frac: UnitInterval = field(default_factory=lambda: parse_unit_interval(0.1))
    k: NonNegativeFloat = field(default_factory=lambda: parse_non_negative_float(10.0))

    def __post_init__(self) -> None:
        if not isinstance(self.shape, LambdaScheduleShape):
            raise TypeError("shape must be refined as LambdaScheduleShape")
        _require_phantom(self.max, NonNegativeFloat, "max")
        _require_phantom(self.warmup_frac, UnitInterval, "warmup_frac")
        _require_phantom(self.k, NonNegativeFloat, "k")
        if self.shape is LambdaScheduleShape.DANN and self.k == 0.0:
            raise ValueError("DANN schedule k must be positive")

    @classmethod
    @beartype
    def from_raw(
        cls,
        *,
        shape: str = "dann",
        max: float | int | str = 1.0,
        warmup_frac: float | int | str = 0.1,
        k: float | int | str = 10.0,
    ) -> LambdaScheduleConfig:
        return cls(
            shape=LambdaScheduleShape(shape),
            max=parse_non_negative_float(max),
            warmup_frac=parse_unit_interval(warmup_frac),
            k=parse_non_negative_float(k),
        )


@dataclass(frozen=True, slots=True)
class KLAnchorConfig:
    """All-or-nothing reference-model KL anchor."""

    reference_model_id: str
    gamma: NonNegativeFloat

    def __post_init__(self) -> None:
        _require_nonempty(self.reference_model_id, "reference_model_id")
        _require_phantom(self.gamma, NonNegativeFloat, "gamma")

    @classmethod
    @beartype
    def from_raw(
        cls,
        *,
        reference_model_id: str,
        gamma: float | int | str,
    ) -> KLAnchorConfig:
        return cls(
            reference_model_id=reference_model_id,
            gamma=parse_non_negative_float(gamma),
        )


@dataclass(frozen=True, slots=True)
class ConcentrationTrainConfig:
    """Shared concentration-training hyperparameters used by later phases."""

    alpha: NonNegativeFloat = field(default_factory=lambda: parse_non_negative_float(1.0))
    lambda_schedule: LambdaScheduleConfig = field(default_factory=LambdaScheduleConfig)
    detach_basis: bool = True
    seed: Seed = field(default_factory=lambda: parse_seed(0))
    policy_lr: NonNegativeFloat = field(
        default_factory=lambda: parse_non_negative_float(1.0e-5)
    )
    reward_head_lr: NonNegativeFloat = field(
        default_factory=lambda: parse_non_negative_float(1.0e-3)
    )
    adversary_lr: NonNegativeFloat = field(
        default_factory=lambda: parse_non_negative_float(1.0e-3)
    )
    policy_weight_decay: NonNegativeFloat = field(
        default_factory=lambda: parse_non_negative_float(0.01)
    )
    head_weight_decay: NonNegativeFloat = field(
        default_factory=lambda: parse_non_negative_float(0.0)
    )
    steps: Rank = field(default_factory=lambda: parse_rank(1000))
    grad_clip: NonNegativeFloat = field(default_factory=lambda: parse_non_negative_float(1.0))
    alternating_minmax: bool = False
    adversary_steps: Rank = field(default_factory=lambda: parse_rank(1))
    kl_anchor: KLAnchorConfig | None = None

    def __post_init__(self) -> None:
        _require_phantom(self.alpha, NonNegativeFloat, "alpha")
        if not isinstance(self.lambda_schedule, LambdaScheduleConfig):
            raise TypeError("lambda_schedule must be LambdaScheduleConfig")
        if type(self.detach_basis) is not bool:
            raise TypeError("detach_basis must be bool")
        _require_phantom(self.seed, Seed, "seed")
        for name in (
            "policy_lr",
            "reward_head_lr",
            "adversary_lr",
            "policy_weight_decay",
            "head_weight_decay",
            "grad_clip",
        ):
            _require_phantom(getattr(self, name), NonNegativeFloat, name)
        if min(self.policy_lr, self.reward_head_lr, self.adversary_lr, self.grad_clip) <= 0.0:
            raise ValueError("learning rates and grad_clip must be positive")
        _require_phantom(self.steps, Rank, "steps")
        if type(self.alternating_minmax) is not bool:
            raise TypeError("alternating_minmax must be bool")
        _require_phantom(self.adversary_steps, Rank, "adversary_steps")
        if self.kl_anchor is not None and not isinstance(self.kl_anchor, KLAnchorConfig):
            raise TypeError("kl_anchor must be KLAnchorConfig or None")

    @classmethod
    @beartype
    def from_raw(
        cls,
        *,
        alpha: float | int | str = 1.0,
        lambda_schedule: LambdaScheduleConfig | None = None,
        detach_basis: bool = True,
        seed: int | str = 0,
        policy_lr: float | int | str = 1.0e-5,
        reward_head_lr: float | int | str = 1.0e-3,
        adversary_lr: float | int | str = 1.0e-3,
        policy_weight_decay: float | int | str = 0.01,
        head_weight_decay: float | int | str = 0.0,
        steps: int | str = 1000,
        grad_clip: float | int | str = 1.0,
        alternating_minmax: bool = False,
        adversary_steps: int | str = 1,
        kl_anchor: KLAnchorConfig | None = None,
    ) -> ConcentrationTrainConfig:
        return cls(
            alpha=parse_non_negative_float(alpha),
            lambda_schedule=lambda_schedule or LambdaScheduleConfig(),
            detach_basis=detach_basis,
            seed=parse_seed(seed),
            policy_lr=parse_non_negative_float(policy_lr),
            reward_head_lr=parse_non_negative_float(reward_head_lr),
            adversary_lr=parse_non_negative_float(adversary_lr),
            policy_weight_decay=parse_non_negative_float(policy_weight_decay),
            head_weight_decay=parse_non_negative_float(head_weight_decay),
            steps=parse_rank(steps),
            grad_clip=parse_non_negative_float(grad_clip),
            alternating_minmax=alternating_minmax,
            adversary_steps=parse_rank(adversary_steps),
            kl_anchor=kl_anchor,
        )


@dataclass(frozen=True, slots=True)
class TrackAConfig:
    """Track-A algorithm selection and seed."""

    algo: TrackAAlgorithm = TrackAAlgorithm.PPO
    seed: Seed = field(default_factory=lambda: parse_seed(0))

    def __post_init__(self) -> None:
        if not isinstance(self.algo, TrackAAlgorithm):
            raise TypeError("algo must be refined as TrackAAlgorithm")
        _require_phantom(self.seed, Seed, "seed")

    @classmethod
    @beartype
    def from_raw(cls, *, algo: str = "ppo", seed: int | str = 0) -> TrackAConfig:
        return cls(algo=TrackAAlgorithm(algo), seed=parse_seed(seed))


@dataclass(frozen=True, slots=True)
class WandbConfig:
    """Weights & Biases mode and project selection."""

    mode: WandbMode = WandbMode.ONLINE
    project: str = "reward-concentration"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, WandbMode):
            raise TypeError("mode must be refined as WandbMode")
        _require_nonempty(self.project, "project")

    @classmethod
    @beartype
    def from_raw(
        cls,
        *,
        mode: str = "online",
        project: str = "reward-concentration",
    ) -> WandbConfig:
        return cls(mode=WandbMode(mode), project=project)
