from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Any, cast

import pytest

from concentration.config import (
    DEFAULT_REWARD_MODEL_ID,
    DEV_POLICY_MODEL_ID,
    DEV_REWARD_MODEL_ID,
    MAIN_POLICY_MODEL_ID,
    AdvHeadConfig,
    AlternatingMinmaxConfig,
    ConcentrationTrainConfig,
    DataConfig,
    KLAnchorConfig,
    LambdaScheduleConfig,
    LambdaScheduleShape,
    ModelConfig,
    ModelDType,
    Nonlinearity,
    Pooling,
    RepExtractionConfig,
    RewardHeadConfig,
    RewardHeadInit,
    RewardHeadMode,
    RewardModelConfig,
    TrackAAlgorithm,
    TrackAConfig,
    WandbConfig,
    WandbMode,
)
from concentration.types import NonNegativeFloat, Rank, Seed, UnitInterval

CONFIG_CLASSES = (
    ModelConfig,
    RewardModelConfig,
    DataConfig,
    RepExtractionConfig,
    RewardHeadConfig,
    AdvHeadConfig,
    LambdaScheduleConfig,
    KLAnchorConfig,
    ConcentrationTrainConfig,
    TrackAConfig,
    WandbConfig,
)


def test_verified_hub_ids_are_exact() -> None:
    assert DEV_POLICY_MODEL_ID == "Qwen/Qwen3-0.6B-Base"
    assert MAIN_POLICY_MODEL_ID == "Qwen/Qwen3-1.7B-Base"
    assert DEFAULT_REWARD_MODEL_ID == "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
    assert DEV_REWARD_MODEL_ID == "Skywork/Skywork-Reward-V2-Qwen3-0.6B"


@pytest.mark.parametrize("config_class", CONFIG_CLASSES)
def test_every_config_dataclass_is_frozen(config_class: type[object]) -> None:
    assert is_dataclass(config_class)
    assert config_class.__dataclass_params__.frozen  # ty: ignore[unresolved-attribute]


def test_model_config_parses_dtype_and_validates_text_fields() -> None:
    config = ModelConfig.from_raw(dtype="float32", device="cpu")
    assert config.model_id == DEV_POLICY_MODEL_ID
    assert config.dtype is ModelDType.FLOAT32
    with pytest.raises(FrozenInstanceError):
        config.device = "cuda"  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="model_id"):
        ModelConfig.from_raw(model_id=" ")
    with pytest.raises(ValueError):
        ModelConfig.from_raw(dtype="float16")
    with pytest.raises(TypeError, match="dtype"):
        ModelConfig(dtype=cast(Any, "float32"))


def test_reward_model_config_refines_counts() -> None:
    config = RewardModelConfig.from_raw(
        model_id=DEV_REWARD_MODEL_ID,
        dtype="float32",
        device="cpu",
        batch_size="2",
        max_length="64",
    )
    assert isinstance(config.batch_size, Rank)
    assert config.batch_size == 2
    assert config.max_length == 64
    with pytest.raises((TypeError, ValueError)):
        RewardModelConfig.from_raw(batch_size=0)
    with pytest.raises(ValueError, match="revision"):
        RewardModelConfig.from_raw(revision="")


def test_data_config_refines_values_and_enforces_nonempty_train_fraction() -> None:
    config = DataConfig.from_raw(
        max_len="128",
        heldout_probe_train_frac="0.2",
        heldout_probe_test_frac=0.3,
        seed="9",
    )
    assert isinstance(config.max_len, Rank)
    assert isinstance(config.heldout_probe_train_frac, UnitInterval)
    assert isinstance(config.seed, Seed)
    assert config.heldout_probe_train_frac + config.heldout_probe_test_frac == 0.5
    with pytest.raises(ValueError, match="sum"):
        DataConfig.from_raw(
            heldout_probe_train_frac=0.5,
            heldout_probe_test_frac=0.5,
        )
    with pytest.raises(ValueError, match="preference_dataset_id"):
        DataConfig.from_raw(preference_dataset_id="")
    with pytest.raises(TypeError, match="max_len"):
        DataConfig(max_len=cast(Any, object()))


@pytest.mark.parametrize("pooling", ["mean", "last", "max", "min"])
def test_rep_extraction_config_accepts_every_pooling(pooling: str) -> None:
    config = RepExtractionConfig.from_raw(layer="3", pooling=pooling)
    assert config.layer == 3
    assert config.pooling is Pooling(pooling)


@pytest.mark.parametrize(("layer", "error"), [(-1, ValueError), (True, TypeError)])
def test_rep_extraction_config_rejects_invalid_layers(
    layer: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        RepExtractionConfig.from_raw(layer=layer)


def test_reward_and_adversary_head_configs_parse_all_fields() -> None:
    reward = RewardHeadConfig.from_raw(
        rank_n="4",
        nonlinearity="sigmoid",
        mode="mlp",
        init="probe",
        trainable=False,
    )
    assert reward.rank_n == 4
    assert reward.nonlinearity is Nonlinearity.SIGMOID
    assert reward.mode is RewardHeadMode.MLP
    assert reward.init is RewardHeadInit.PROBE
    assert reward.trainable is False

    adversary = AdvHeadConfig.from_raw(hidden_dims=("8", 4), nonlinearity="sigmoid")
    assert adversary.hidden_dims == (8, 4)
    assert all(isinstance(dimension, Rank) for dimension in adversary.hidden_dims)
    assert adversary.nonlinearity is Nonlinearity.SIGMOID


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (RewardHeadConfig.from_raw, {"rank_n": 0}),
        (RewardHeadConfig.from_raw, {"nonlinearity": "gelu"}),
        (RewardHeadConfig.from_raw, {"mode": "linear"}),
        (RewardHeadConfig.from_raw, {"init": "random"}),
        (AdvHeadConfig.from_raw, {"hidden_dims": (0,)}),
        (AdvHeadConfig.from_raw, {"nonlinearity": "gelu"}),
    ],
)
def test_head_configs_crash_on_invalid_values(
    builder: Callable[..., object],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        builder(**kwargs)


@pytest.mark.parametrize("shape", ["dann", "linear", "constant", "cosine"])
def test_lambda_schedule_config_accepts_all_shapes(shape: str) -> None:
    config = LambdaScheduleConfig.from_raw(shape=shape, max="3", warmup_frac="0.2", k=5)
    assert config.shape is LambdaScheduleShape(shape)
    assert isinstance(config.max, NonNegativeFloat)
    assert config.max == 3.0
    assert config.warmup_frac == 0.2
    assert config.k == 5.0


def test_lambda_schedule_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        LambdaScheduleConfig.from_raw(shape="quadratic")
    with pytest.raises((TypeError, ValueError)):
        LambdaScheduleConfig.from_raw(max=-1)
    with pytest.raises((TypeError, ValueError)):
        LambdaScheduleConfig.from_raw(warmup_frac=1.1)
    with pytest.raises(ValueError, match="k"):
        LambdaScheduleConfig.from_raw(shape="dann", k=0)


def test_kl_anchor_is_an_all_or_nothing_bundle() -> None:
    anchor = KLAnchorConfig.from_raw(reference_model_id=MAIN_POLICY_MODEL_ID, gamma="0.1")
    without_anchor = ConcentrationTrainConfig.from_raw()
    with_anchor = ConcentrationTrainConfig.from_raw(kl_anchor=anchor)
    assert without_anchor.kl_anchor is None
    assert with_anchor.kl_anchor is anchor
    with pytest.raises(TypeError, match="kl_anchor"):
        ConcentrationTrainConfig(kl_anchor=cast(Any, {"gamma": 0.1}))
    with pytest.raises(ValueError, match="reference_model_id"):
        KLAnchorConfig.from_raw(reference_model_id="", gamma=0.1)


def test_concentration_train_config_refines_every_numeric_boundary() -> None:
    schedule = LambdaScheduleConfig.from_raw(shape="cosine", max=2, warmup_frac=0.25, k=3)
    config = ConcentrationTrainConfig.from_raw(
        alpha="2",
        lambda_schedule=schedule,
        detach_basis=False,
        seed="12",
        policy_lr="0.001",
        reward_head_lr="0.002",
        adversary_lr="0.003",
        policy_weight_decay="0.01",
        head_weight_decay=0,
        steps="25",
        grad_clip="0.5",
        alternating=AlternatingMinmaxConfig.from_raw(adversary_steps="2"),
    )
    for name in (
        "alpha",
        "policy_lr",
        "reward_head_lr",
        "adversary_lr",
        "policy_weight_decay",
        "head_weight_decay",
        "grad_clip",
    ):
        assert isinstance(getattr(config, name), NonNegativeFloat)
    assert config.lambda_schedule is schedule
    assert config.seed == 12
    assert config.steps == 25
    assert config.detach_basis is False
    assert config.alternating is not None
    assert config.alternating.adversary_steps == 2


def test_alternating_minmax_bundle_is_all_or_nothing() -> None:
    assert ConcentrationTrainConfig().alternating is None
    bundle = AlternatingMinmaxConfig.from_raw(adversary_steps=3)
    assert ConcentrationTrainConfig(alternating=bundle).alternating is bundle
    assert bundle.adversary_steps == 3
    with pytest.raises(TypeError, match="adversary_steps"):
        AlternatingMinmaxConfig(adversary_steps=cast(Any, 0))
    with pytest.raises(FrozenInstanceError):
        cast(Any, bundle).adversary_steps = 4
    with pytest.raises(TypeError, match="AlternatingMinmaxConfig"):
        ConcentrationTrainConfig(alternating=cast(Any, True))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -1},
        {"policy_lr": 0},
        {"reward_head_lr": 0},
        {"adversary_lr": 0},
        {"grad_clip": 0},
        {"steps": 0},
        {"adversary_steps": 0},
    ],
)
def test_concentration_train_config_crashes_on_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ConcentrationTrainConfig.from_raw(**cast(Any, kwargs))


@pytest.mark.parametrize("algo", ["ppo", "grpo"])
def test_track_a_config_accepts_supported_algorithms(algo: str) -> None:
    config = TrackAConfig.from_raw(algo=algo, seed="4")
    assert config.algo is TrackAAlgorithm(algo)
    assert config.seed == 4


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_wandb_config_accepts_supported_modes(mode: str) -> None:
    config = WandbConfig.from_raw(mode=mode, project="study")
    assert config.mode is WandbMode(mode)
    assert config.project == "study"


def test_track_a_and_wandb_configs_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        TrackAConfig.from_raw(algo="dpo")
    with pytest.raises(ValueError):
        WandbConfig.from_raw(mode="local")
    with pytest.raises(ValueError, match="project"):
        WandbConfig.from_raw(project=" ")


def test_config_surface_contains_only_declared_fields() -> None:
    assert [item.name for item in fields(RepExtractionConfig)] == ["layer", "pooling"]
    assert [item.name for item in fields(RewardHeadConfig)] == [
        "rank_n",
        "nonlinearity",
        "mode",
        "init",
        "trainable",
    ]
    assert [item.name for item in fields(AdvHeadConfig)] == ["hidden_dims", "nonlinearity"]
    assert [item.name for item in fields(LambdaScheduleConfig)] == [
        "shape",
        "max",
        "warmup_frac",
        "k",
    ]


@pytest.mark.parametrize(
    "build",
    [
        lambda: RewardModelConfig(dtype=cast(Any, "float32")),
        lambda: RepExtractionConfig(layer=0, pooling=cast(Any, "mean")),
        lambda: RewardHeadConfig(nonlinearity=cast(Any, "relu")),
        lambda: RewardHeadConfig(mode=cast(Any, "mlp")),
        lambda: RewardHeadConfig(init=cast(Any, "probe")),
        lambda: RewardHeadConfig(trainable=cast(Any, 1)),
        lambda: AdvHeadConfig(hidden_dims=cast(Any, [1])),
        lambda: AdvHeadConfig(nonlinearity=cast(Any, "relu")),
        lambda: LambdaScheduleConfig(shape=cast(Any, "dann")),
        lambda: ConcentrationTrainConfig(lambda_schedule=cast(Any, "dann")),
        lambda: ConcentrationTrainConfig(detach_basis=cast(Any, 1)),
        lambda: ConcentrationTrainConfig(alternating=cast(Any, 1)),
        lambda: TrackAConfig(algo=cast(Any, "ppo")),
        lambda: WandbConfig(mode=cast(Any, "online")),
    ],
)
def test_direct_config_construction_rejects_unrefined_field_types(
    build: Callable[[], object],
) -> None:
    with pytest.raises(TypeError):
        build()
