import subprocess
import tomllib
from pathlib import Path
from typing import Annotated, Literal

import pytest
import tomli_w
from pydantic import BaseModel, Field, ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.shared import SlurmConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.configs.value import ClassificationValueLossConfig, ValueFunctionConfig, ValueReplayConfig
from prime_rl.entrypoints.rl import validate_value_tokenizer_compatibility, write_slurm_script
from prime_rl.utils.config import BaseConfig, cli

# All config config classes
CONFIG_CLASSES = [
    RLConfig,
    TrainerConfig,
    SFTConfig,
    OrchestratorConfig,
    InferenceConfig,
]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`."""
    config_files = list(Path("configs").rglob("*.toml"))
    example_files = list(Path("examples").rglob("*.toml"))

    return config_files + example_files


def is_eval_config(path: Path) -> bool:
    """vf-eval TOMLs live under configs but are not prime-rl entrypoint configs."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    return isinstance(data.get("eval"), list)


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path):
    """Tests that all config files can be loaded by at least one config class."""
    if is_eval_config(config_file):
        pytest.skip("vf-eval TOML files are not prime-rl entrypoint configs")

    could_parse = []
    for config_cls in CONFIG_CLASSES:
        try:
            cli(config_cls, args=["@", config_file.as_posix()])
            could_parse.append(True)
        except (ValidationError, ConfigFileError, SystemExit):
            could_parse.append(False)
    assert any(could_parse), f"No config class could be parsed from {config_file}"


class NestedConfig(BaseConfig):
    lr: float = 1e-4
    weight_decay: float = 0.01
    name: str = "default"


class VariantA(BaseModel):
    type: Literal["a"] = "a"
    alpha: float = 0.1
    shared: int = 1


class VariantB(BaseModel):
    type: Literal["b"] = "b"
    beta: float = 0.2
    shared: int = 1


VariantType = Annotated[VariantA | VariantB, Field(discriminator="type")]


class DummyConfig(BaseConfig):
    name: str = "experiment"
    seed: int = 42
    nested: NestedConfig = NestedConfig()
    variant: VariantType = VariantA()


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_defaults():
    """All defaults are applied when no TOML or CLI args are given."""
    config = cli(DummyConfig, args=[])
    assert config.name == "experiment"
    assert config.seed == 42
    assert config.nested.lr == 1e-4
    assert config.nested.weight_decay == 0.01
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.1


def test_toml_partial_nested_override(tmp_path):
    """Partially overriding a nested model preserves unset field defaults."""
    write_toml(tmp_path / "cfg.toml", {"nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.nested.lr == 3e-4
    assert config.nested.weight_decay == 0.01
    assert config.nested.name == "default"


def test_toml_discriminated_union_default_type(tmp_path):
    """Overriding a discriminated union field without 'type' uses the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"alpha": 0.9}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.9
    assert config.variant.shared == 1


def test_toml_discriminated_union_switch_variant(tmp_path):
    """Providing an explicit 'type' switches to that variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b"}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.2


def test_toml_discriminated_union_override_switch_variant(tmp_path):
    """Providing an explicit 'type' overrides the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b", "beta": 0.5}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.5


def test_cli_overrides_defaults():
    """CLI args override defaults."""
    config = cli(DummyConfig, args=["--name", "my-run", "--seed", "7"])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 1e-4


def test_toml_overrides_defaults(tmp_path):
    """TOML overrides defaults."""
    write_toml(tmp_path / "cfg.toml", {"name": "my-run", "seed": 7, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 3e-4


def test_cli_overrides_toml(tmp_path):
    """CLI args override TOML."""
    write_toml(tmp_path / "cfg.toml", {"seed": 1, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml"), "--seed", "99", "--nested.lr", "5e-5"])
    assert config.seed == 99
    assert config.nested.lr == 5e-5
    # TOML value not overridden by CLI should still be applied (not reverted to class default)
    assert config.nested.weight_decay == 0.01


def test_removed_fused_lm_head_chunk_size_field_is_rejected():
    with pytest.raises(ValidationError, match="fused_lm_head_chunk_size"):
        TrainerModelConfig.model_validate({"fused_lm_head_chunk_size": "auto"})


def test_env_algo_overrides_top_level():
    config = OrchestratorConfig.model_validate(
        {
            "renderer": {"name": "qwen3"},  # echo needs the renderer's role attribution
            "algo": {"type": "echo"},
            "train": {"env": [{"id": "a", "algo": {"type": "grpo"}}, {"id": "b"}]},
        }
    )
    env_a, env_b = config.train.env
    # Env a sets its own algorithm; only env b inherits the top-level echo algorithm.
    assert env_a.algo is not None and env_a.algo.type == "grpo"
    assert env_b.algo is not None and env_b.algo.type == "echo"

    # Resolved configs round-trip.
    dumped = config.model_dump(exclude_none=True)
    reloaded = OrchestratorConfig.model_validate(dumped)
    assert reloaded.train.env[0].algo is not None and reloaded.train.env[0].algo.type == "grpo"


def test_trainer_enable_token_export_cli_flag():
    assert not cli(TrainerConfig, args=[]).enable_token_export
    assert cli(TrainerConfig, args=["--enable-token-export"]).enable_token_export


def test_single_node_auto_inference_client_dp_rank_count_matches_local_dp():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"parallel": {"tp": 1}},
            "deployment": {
                "type": "single_node",
                "gpus_per_node": 4,
                "num_train_gpus": 2,
                "num_infer_gpus": 2,
            },
        }
    )

    assert config.inference is not None
    assert config.inference.parallel.dp == 2
    assert config.orchestrator.model.client.dp_rank_count == 2


def test_value_baseline_requires_value_function():
    with pytest.raises(ValidationError, match="value-backed baselines require"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {"algo": {"type": "grpo", "baseline": {"type": "value"}}},
            }
        )


def test_value_function_defaults_omitted_grpo_baselines_to_value():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {
                "group_size": 2,
                "algo": {"type": "grpo"},
                "train": {
                    "env": [
                        {"id": "inherited"},
                        {"id": "implicit", "algo": {"type": "grpo"}},
                        {
                            "id": "explicit",
                            "algo": {"type": "grpo", "baseline": {"type": "mean"}},
                        },
                    ]
                },
            },
            "value_function": {},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )

    assert config.orchestrator.algo.baseline.type == "value"
    inherited, implicit, explicit = config.orchestrator.train.env
    assert inherited.algo is not None and inherited.algo.baseline.type == "value"
    assert implicit.algo is not None and implicit.algo.baseline.type == "value"
    assert explicit.algo is not None and explicit.algo.baseline.type == "mean"
    assert config.value_function is not None and config.value_function.warmup_updates == 1
    assert config.orchestrator.value_function is not None
    assert config.orchestrator.value_function.warmup_updates == 1


def test_grpo_without_value_function_keeps_mean_baseline():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {"algo": {"type": "grpo"}},
        }
    )

    assert config.orchestrator.algo.baseline.type == "mean"


def test_value_function_preserves_explicit_mean_baseline_and_length_penalty():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {
                "algo": {
                    "type": "grpo",
                    "baseline": {"type": "mean"},
                    "length_penalty": {},
                }
            },
            "value_function": {},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )

    assert config.orchestrator.algo.baseline.type == "mean"
    assert config.orchestrator.algo.length_penalty is not None
    assert config.value_function is not None and config.value_function.warmup_updates == 0


@pytest.mark.parametrize("warmup_updates", [0, 3])
def test_value_function_preserves_explicit_warmup(warmup_updates):
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {"algo": {"type": "grpo"}},
            "value_function": {"warmup_updates": warmup_updates},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )

    assert config.value_function is not None
    assert config.value_function.warmup_updates == warmup_updates
    assert config.orchestrator.value_function is not None
    assert config.orchestrator.value_function.warmup_updates == warmup_updates


def test_implicit_value_baseline_rejects_length_penalty():
    with pytest.raises(ValidationError, match="value-backed GRPO baselines cannot be combined"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "algo": {"type": "grpo", "length_penalty": {}},
                },
                "value_function": {},
                "deployment": {"type": "single_node", "gpus_per_node": 4},
            }
        )


def test_value_function_resolves_separate_model_and_gpu_roles():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {
                "group_size": 2,
                "algo": {"type": "grpo", "baseline": {"type": "value"}},
            },
            "value_function": {},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )

    assert config.value_function is not None
    assert config.value_function.model is not None
    assert config.value_function.model.name == config.trainer.model.name
    assert config.orchestrator.value_function is not None
    assert config.orchestrator.algo.baseline.type == "value"
    assert config.value_function.batch_size == config.orchestrator.batch_size == 128
    assert config.value_function.evaluator.placement == "dedicated"
    assert config.value_function.evaluator.dtype == "bfloat16"
    assert config.deployment.type == "single_node"
    assert config.deployment.num_value_train_gpus == 1
    assert config.deployment.num_value_eval_gpus == 1


def test_trainer_placed_value_evaluator_uses_no_extra_gpu():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {"algo": {"type": "grpo"}},
            "value_function": {"evaluator": {"placement": "trainer"}},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )

    assert config.value_function is not None
    assert config.value_function.evaluator.base_url == ["http://127.0.0.1:29612"]
    assert config.orchestrator.value_function is not None
    assert config.orchestrator.value_function.evaluator.placement == "trainer"
    assert config.deployment.type == "single_node"
    assert config.deployment.num_value_train_gpus == 1
    assert config.deployment.num_value_eval_gpus == 0


def test_trainer_placed_value_evaluator_rejects_dedicated_gpu():
    with pytest.raises(ValidationError, match="requires deployment.num_value_eval_gpus=0"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {"algo": {"type": "grpo"}},
                "value_function": {"evaluator": {"placement": "trainer"}},
                "deployment": {
                    "type": "single_node",
                    "gpus_per_node": 4,
                    "num_value_eval_gpus": 1,
                },
            }
        )


def test_multi_node_trainer_placed_value_evaluator_rejects_dedicated_node():
    with pytest.raises(ValidationError, match="requires deployment.num_value_eval_nodes=0"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {"algo": {"type": "grpo"}},
                "value_function": {"evaluator": {"placement": "trainer"}},
                "deployment": {
                    "type": "multi_node",
                    "gpus_per_node": 4,
                    "num_train_nodes": 1,
                    "num_value_eval_nodes": 1,
                },
            }
        )


@pytest.mark.parametrize(
    ("evaluator", "weight_broadcast", "message"),
    [
        (
            {
                "placement": "trainer",
                "base_url": ["http://127.0.0.1:29612", "http://127.0.0.1:29614"],
            },
            {},
            "exactly one evaluator base_url",
        ),
        (
            {"placement": "trainer", "base_url": ["http://127.0.0.1:29614"]},
            {},
            "base_url port must equal",
        ),
        (
            {"placement": "trainer"},
            {"evaluator_world_size": 2},
            "exactly one evaluator endpoint",
        ),
    ],
)
def test_trainer_placed_value_evaluator_requires_one_endpoint(evaluator, weight_broadcast, message):
    with pytest.raises(ValidationError, match=message):
        ValueFunctionConfig.model_validate(
            {
                "evaluator": evaluator,
                "weight_broadcast": weight_broadcast,
            }
        )


def test_dedicated_value_evaluator_requires_one_url_per_replica():
    with pytest.raises(ValidationError, match="base_url count"):
        ValueFunctionConfig.model_validate(
            {
                "evaluator": {"base_url": ["http://127.0.0.1:29612"]},
                "weight_broadcast": {"evaluator_world_size": 2},
            }
        )


def test_trainer_placed_value_evaluator_ignores_unused_weight_ports():
    config = ValueFunctionConfig.model_validate(
        {
            "evaluator": {"placement": "trainer"},
            "weight_broadcast": {
                "port": 29610,
                "control_port": 29612,
            },
        }
    )

    assert config.evaluator.placement == "trainer"


@pytest.mark.parametrize(
    "name",
    ["PRIME_RL_RUN_DONE_FILE", "VALUE_EVALUATOR_PORT", "VALUE_TRAIN_MASTER", "VALUE_TRAIN_MASTER_PORT"],
)
def test_value_function_rejects_launcher_managed_env_vars(name):
    with pytest.raises(ValidationError, match="launcher-managed"):
        ValueFunctionConfig.model_validate({"env_vars": {name: "invalid"}})


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ({"dp_replicate": 2}, "dp_replicate"),
        ({"fsdp_cpu_offload": True, "optim_cpu_offload": False}, "FSDP CPU offload"),
        ({"fp8": True}, "FP8"),
        ({"ep": 2}, "expert parallelism"),
        ({"ep": "auto", "ep_comm_backend": "deepep"}, "DeepEP"),
    ],
)
def test_trainer_placed_value_evaluator_rejects_unsupported_topology(model, message):
    with pytest.raises(ValidationError, match=message):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {"algo": {"type": "grpo"}},
                "value_function": {
                    "model": model,
                    "evaluator": {"placement": "trainer"},
                },
                "deployment": {"type": "single_node", "gpus_per_node": 4},
            }
        )


def test_value_evaluator_batch_ceiling_may_be_smaller_than_sequence_length():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {"algo": {"type": "grpo"}},
            "value_function": {"evaluator": {"max_batch_tokens": 128}},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )

    assert config.value_function is not None
    assert config.value_function.model is not None
    assert config.value_function.evaluator.max_batch_tokens < config.value_function.model.seq_len


def test_shared_observability_reaches_value_trainer():
    config = RLConfig.model_validate(
        {
            "log": {"level": "debug", "json_logging": True},
            "wandb": {"project": "critic-test", "name": "shared-run"},
            "trainer": {},
            "orchestrator": {"algo": {"type": "grpo"}},
            "value_function": {},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )

    assert config.value_function is not None
    assert config.value_function.log.level == "debug"
    assert config.value_function.log.json_logging is True
    assert config.value_function.wandb is not None
    assert config.value_function.wandb.project == "critic-test"
    assert config.value_function.wandb.name == "shared-run"


def test_value_function_defaults_to_binary_classification_and_independent_lambdas():
    config = ValueFunctionConfig()

    assert isinstance(config.loss, ClassificationValueLossConfig)
    assert config.loss.num_bins == 2
    assert config.loss.reward_range == (0.0, 1.0)
    assert config.optim.lr == 1e-5
    assert config.replay.max_updates_per_rollout == 1
    assert config.replay.capacity is None
    assert config.replay.refill_size is None
    assert config.replay.seed == 0
    assert config.transport.max_pending_rollouts == 2048
    assert config.gae_lambda == 1.0
    assert config.value_target_lambda == 1.0
    assert config.warmup_updates == 0


def test_default_value_replay_matches_one_optimizer_batch():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {"batch_size": 16, "algo": {"type": "grpo"}},
            "value_function": {},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )

    assert config.value_function is not None
    assert config.value_function.batch_size == config.orchestrator.batch_size == 16
    assert config.value_function.replay.max_updates_per_rollout == 1
    assert config.value_function.replay.capacity == 16
    assert config.value_function.replay.refill_size == 16
    assert config.orchestrator.value_function is not None
    assert config.orchestrator.value_function.replay == config.value_function.replay


def test_legacy_value_updates_per_batch_migrates_to_replay():
    with pytest.warns(FutureWarning, match="updates_per_batch is deprecated"):
        config = ValueFunctionConfig.model_validate({"batch_size": 8, "updates_per_batch": 3})

    assert config.replay.max_updates_per_rollout == 3
    assert config.replay.capacity == 24
    assert config.replay.refill_size == 24


def test_legacy_value_updates_per_batch_rejects_conflicting_replay_setting():
    with pytest.raises(ValidationError, match="updates_per_batch conflicts"):
        ValueFunctionConfig.model_validate(
            {
                "updates_per_batch": 2,
                "replay": {"max_updates_per_rollout": 3},
            }
        )


@pytest.mark.parametrize(("legacy", "replacement"), [(3, "03"), ("3.0", 3)])
def test_legacy_value_updates_per_batch_accepts_equivalent_replay_setting(legacy, replacement):
    with pytest.warns(FutureWarning, match="updates_per_batch is deprecated"):
        config = ValueFunctionConfig.model_validate(
            {
                "updates_per_batch": legacy,
                "replay": {"max_updates_per_rollout": replacement},
            }
        )

    assert config.replay.max_updates_per_rollout == 3


def test_legacy_value_updates_per_batch_does_not_overwrite_invalid_replay_setting():
    with pytest.raises(ValidationError, match="max_updates_per_rollout"):
        ValueFunctionConfig.model_validate(
            {
                "updates_per_batch": 3,
                "replay": {"max_updates_per_rollout": None},
            }
        )


def test_legacy_value_updates_per_batch_preserves_explicit_replay_model_fields():
    with pytest.warns(FutureWarning, match="updates_per_batch is deprecated"):
        config = ValueFunctionConfig.model_validate(
            {
                "updates_per_batch": 3,
                "replay": ValueReplayConfig(capacity=32),
            }
        )

    assert config.replay.max_updates_per_rollout == 3
    assert config.replay.capacity == 32


def test_legacy_value_cli_settings_migrate():
    with pytest.warns(FutureWarning) as caught:
        config = cli(
            ValueFunctionConfig,
            args=["--updates-per-batch", "3", "--transport.type", "zmq_latest"],
        )

    assert len(caught) == 2
    assert config.replay.max_updates_per_rollout == 3
    assert config.transport.type == "zmq"


def test_legacy_latest_value_transport_type_migrates_to_zmq():
    with pytest.warns(FutureWarning, match="zmq_latest.*deprecated"):
        config = ValueFunctionConfig.model_validate({"transport": {"type": "zmq_latest"}})

    assert config.transport.type == "zmq"


def test_value_replay_defaults_resolve_from_rollout_batch_size():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {"batch_size": 16, "algo": {"type": "grpo"}},
            "value_function": {"replay": {"max_updates_per_rollout": 3}},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )

    assert config.value_function is not None
    assert config.value_function.replay.capacity == 48
    assert config.value_function.replay.refill_size == 48
    assert config.value_function.replay.resolved_capacity == 48
    assert config.value_function.replay.resolved_refill_size == 48


def test_standalone_value_config_resolves_replay_when_batch_size_is_explicit():
    config = ValueFunctionConfig.model_validate({"batch_size": 8, "replay": {"max_updates_per_rollout": 2}})

    assert config.replay.capacity == 16
    assert config.replay.refill_size == 16


def test_value_replay_explicit_capacity_and_refill_are_preserved():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {"batch_size": 16, "algo": {"type": "grpo"}},
            "value_function": {
                "replay": {
                    "max_updates_per_rollout": 3,
                    "capacity": 64,
                    "refill_size": 32,
                    "seed": 7,
                }
            },
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )

    assert config.value_function is not None
    assert config.value_function.replay.capacity == 64
    assert config.value_function.replay.refill_size == 32
    assert config.value_function.replay.seed == 7


@pytest.mark.parametrize(
    ("replay", "message"),
    [
        ({"capacity": 15}, "capacity must be at least"),
        ({"capacity": 32, "refill_size": 15}, "refill_size must be at least"),
        ({"capacity": 32, "refill_size": 33}, "refill_size cannot exceed"),
    ],
)
def test_value_replay_requires_batch_size_at_most_refill_at_most_capacity(replay, message):
    with pytest.raises(ValidationError, match=message):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {"batch_size": 16, "algo": {"type": "grpo"}},
                "value_function": {"replay": replay},
                "deployment": {"type": "single_node", "gpus_per_node": 4},
            }
        )


def test_token_batched_policy_requires_explicit_value_rollout_batch_size():
    with pytest.raises(ValidationError, match="value_function.batch_size must be set"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "token_batch_size": 4096,
                    "max_inflight_rollouts": 16,
                    "algo": {"type": "grpo"},
                },
                "value_function": {},
                "deployment": {"type": "single_node", "gpus_per_node": 4},
            }
        )

    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {
                "token_batch_size": 4096,
                "max_inflight_rollouts": 16,
                "algo": {"type": "grpo"},
            },
            "value_function": {"batch_size": 32},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )
    assert config.value_function is not None
    assert config.value_function.batch_size == 32


def test_group_only_baseline_does_not_require_value_function():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {
                "group_size": 2,
                "algo": {"type": "grpo", "baseline": {"type": "leave_one_out"}},
            },
        }
    )

    assert config.value_function is None


def test_leave_one_out_requires_siblings_without_value_function():
    with pytest.raises(ValidationError, match="leave_one_out baseline requires group_size >= 2"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "group_size": 1,
                    "algo": {"type": "grpo", "baseline": {"type": "leave_one_out"}},
                },
            }
        )


def test_value_warmup_cannot_outlive_value_trainer():
    with pytest.raises(ValidationError, match="warmup_updates cannot exceed"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {"algo": {"type": "grpo"}},
                "value_function": {"warmup_updates": 3, "max_steps": 2},
                "deployment": {"type": "single_node", "gpus_per_node": 4},
            }
        )


def test_multi_node_auto_inference_client_dp_rank_count_uses_router_url():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"parallel": {"tp": 4}},
            "deployment": {
                "type": "multi_node",
                "gpus_per_node": 8,
                "num_train_nodes": 1,
                "num_infer_nodes": 2,
            },
            "slurm": {},
        }
    )

    assert config.inference is not None
    assert config.inference.data_parallel_size_local == 2
    assert config.inference.parallel.dp == 2
    assert config.orchestrator.model.client.dp_rank_count == 1


def test_value_trainer_supports_multiple_nodes_and_evaluator_replicas():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {
                "group_size": 2,
                "algo": {"type": "grpo", "baseline": {"type": "value"}},
            },
            "inference": {"parallel": {"tp": 4}},
            "value_function": {
                "evaluator": {
                    "base_url": ["http://127.0.0.1:29612", "http://127.0.0.1:29614"],
                },
                "weight_broadcast": {"evaluator_world_size": 2},
            },
            "deployment": {
                "type": "multi_node",
                "gpus_per_node": 4,
                "num_train_nodes": 1,
                "num_infer_nodes": 1,
                "num_value_train_nodes": 2,
                "num_value_eval_nodes": 2,
            },
            "slurm": {},
        }
    )

    assert config.deployment.type == "multi_node"
    assert config.deployment.num_value_train_nodes == 2
    assert config.deployment.num_value_eval_nodes == 2
    assert config.value_function is not None
    assert config.value_function.weight_broadcast.evaluator_world_size == 2


def test_multi_node_value_evaluator_urls_render_as_one_cli_argument(tmp_path):
    config = RLConfig.model_validate(
        {
            "output_dir": str(tmp_path / "output"),
            "trainer": {},
            "orchestrator": {
                "group_size": 2,
                "algo": {"type": "grpo", "baseline": {"type": "value"}},
            },
            "inference": {"parallel": {"tp": 4}},
            "value_function": {
                "evaluator": {
                    "base_url": ["http://127.0.0.1:29612", "http://127.0.0.1:29614"],
                },
                "weight_broadcast": {"evaluator_world_size": 2},
            },
            "deployment": {
                "type": "multi_node",
                "gpus_per_node": 4,
                "num_train_nodes": 1,
                "num_infer_nodes": 1,
                "num_value_train_nodes": 2,
                "num_value_eval_nodes": 2,
            },
            "slurm": {},
        }
    )
    script_path = tmp_path / "rl.sbatch"
    write_slurm_script(config, tmp_path / "configs", script_path)
    script = script_path.read_text()
    assignment = next(line for line in script.splitlines() if line.startswith("VALUE_EVAL_URLS_JSON="))

    assert "uv run value-evaluator" in script
    assert "--weight-broadcast.host $VALUE_TRAIN_MASTER" in script
    subprocess.run(["bash", "-n", str(script_path)], check=True)
    rendered = subprocess.run(
        [
            "bash",
            "-c",
            f'export VALUE_EVAL_URLS="http://eval-0:29612 http://eval-1:29612"; {assignment}; printf %s "$VALUE_EVAL_URLS_JSON"',
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert rendered.stdout == '["http://eval-0:29612","http://eval-1:29612"]'


def test_multi_node_trainer_placed_evaluator_renders_without_evaluator_node(tmp_path):
    config = RLConfig.model_validate(
        {
            "output_dir": str(tmp_path / "output"),
            "trainer": {},
            "orchestrator": {"algo": {"type": "grpo"}},
            "inference": {"parallel": {"tp": 4}},
            "value_function": {"evaluator": {"placement": "trainer"}},
            "deployment": {
                "type": "multi_node",
                "gpus_per_node": 4,
                "num_train_nodes": 1,
                "num_infer_nodes": 1,
                "num_value_train_nodes": 1,
            },
            "slurm": {},
        }
    )
    script_path = tmp_path / "rl.sbatch"
    write_slurm_script(config, tmp_path / "configs", script_path)
    script = script_path.read_text()

    assert config.deployment.type == "multi_node"
    assert config.deployment.num_value_eval_nodes == 0
    assert "#SBATCH --nodes=3" in script
    assert 'VALUE_EVAL_URLS="http://${VALUE_TRAIN_MASTER}:${VALUE_EVALUATOR_PORT}"' in script
    assert "VALUE_EVAL_HOSTS" not in script
    assert "uv run value-evaluator" not in script
    assert "--weight-broadcast.host $VALUE_TRAIN_MASTER" not in script
    subprocess.run(["bash", "-n", str(script_path)], check=True)


@pytest.mark.parametrize(
    ("value_function", "field"),
    [
        (
            {
                "evaluator": {
                    "placement": "trainer",
                    "port": 29510,
                    "base_url": ["http://127.0.0.1:29510"],
                }
            },
            "value_function.evaluator.port",
        ),
        (
            {"evaluator": {"placement": "trainer"}, "transport": {"port": 29510}},
            "value_function.transport.port",
        ),
        ({"transport": {"port": 29510}}, "value_function.transport.port"),
        ({"weight_broadcast": {"port": 29510}}, "value_function.weight_broadcast.port"),
        (
            {"weight_broadcast": {"control_port": 29510}},
            "value_function.weight_broadcast.control_port",
        ),
    ],
)
def test_multi_node_value_services_reject_the_value_trainer_rendezvous_port(value_function, field):
    with pytest.raises(ValidationError, match=field):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {"algo": {"type": "grpo"}},
                "inference": {"parallel": {"tp": 4}},
                "value_function": value_function,
                "deployment": {
                    "type": "multi_node",
                    "gpus_per_node": 4,
                    "num_train_nodes": 1,
                    "num_infer_nodes": 1,
                    "num_value_train_nodes": 1,
                },
                "slurm": {},
            }
        )


def test_value_model_must_cover_orchestrator_sequence_length():
    with pytest.raises(ValidationError, match="value_function.model.seq_len must be at least"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {"seq_len": 4096, "algo": {"type": "grpo"}},
                "value_function": {"model": {"seq_len": 2048}},
                "deployment": {"type": "single_node", "gpus_per_node": 4},
            }
        )


def test_value_model_ep_must_fit_critic_world():
    with pytest.raises(ValidationError, match="model.ep must divide"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {"algo": {"type": "grpo"}},
                "value_function": {"model": {"ep": 3}},
                "deployment": {"type": "single_node", "gpus_per_node": 4},
            }
        )


def test_distinct_value_model_tokenizer_must_match_policy(monkeypatch):
    from transformers import AutoTokenizer

    class FakeTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        unk_token_id = 3

        def __init__(self, vocab):
            self.vocab = vocab

        def get_vocab(self):
            return self.vocab

    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {"algo": {"type": "grpo"}},
            "value_function": {
                "model": {"name": "example/value-model"},
                "tokenizer_name": "example/value-tokenizer",
            },
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )
    tokenizers = iter([FakeTokenizer({"policy": 0}), FakeTokenizer({"critic": 0})])
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: next(tokenizers))

    with pytest.raises(ValueError, match="tokenizer is incompatible"):
        validate_value_tokenizer_compatibility(config)


def test_slurm_scheduler_selection_fields_are_preserved():
    config = SlurmConfig.model_validate(
        {
            "partition": "",
            "constraint": "gpu",
            "qos": "standard",
        }
    )

    assert config.template_vars["partition"] == ""
    assert config.template_vars["constraint"] == "gpu"
    assert config.template_vars["qos"] == "standard"


def test_orchestrator_vlm_requires_renderer():
    with pytest.raises(ValidationError, match="renderer"):
        OrchestratorConfig.model_validate(
            {
                "model": {
                    "name": "Qwen/Qwen3-VL-4B-Instruct",
                    "vlm": {
                        "vision_encoder_attr": "model.visual",
                        "language_model_attr": "model.language_model",
                    },
                },
                "renderer": None,
            }
        )

    config = OrchestratorConfig.model_validate(
        {
            "model": {
                "name": "Qwen/Qwen3-VL-4B-Instruct",
                "vlm": {
                    "vision_encoder_attr": "model.visual",
                    "language_model_attr": "model.language_model",
                },
            },
        }
    )

    assert config.renderer is not None


def test_selective_activation_checkpointing_requires_custom_impl():
    with pytest.raises(ValidationError, match="Selective activation checkpointing requires model.impl='custom'"):
        TrainerModelConfig.model_validate({"impl": "hf", "ac": {"mode": "selective"}})


def test_shared_model_name_propagates_to_subconfigs():
    model_name = "PrimeIntellect/test-model"
    config = RLConfig.model_validate(
        {
            "model": {"name": model_name},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {},
        }
    )
    assert config.trainer.model.name == model_name
    assert config.orchestrator.model.name == model_name
    assert config.inference is not None and config.inference.model.name == model_name
    assert config.trainer.tokenizer.name == model_name
    assert config.orchestrator.tokenizer.name == model_name


def test_shared_tokenizer_propagates_when_subconfigs_unset():
    config = RLConfig.model_validate(
        {
            "model": {"name": "my-model"},
            "tokenizer": {"name": "my-tokenizer"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.tokenizer.name == "my-tokenizer"
    assert config.orchestrator.tokenizer.name == "my-tokenizer"


def test_shared_and_sub_tokenizer_name_conflict_raises():
    """Setting tokenizer.name in both [tokenizer] and [trainer.tokenizer]
    is a config conflict — the sub-config would silently win, and any later
    CLI override of [tokenizer].name would silently no-op for the trainer."""
    with pytest.raises(ValidationError, match=r"tokenizer.name.*trainer.tokenizer.name"):
        RLConfig.model_validate(
            {
                "model": {"name": "my-model"},
                "tokenizer": {"name": "shared-tok"},
                "trainer": {"tokenizer": {"name": "trainer-tok"}},
                "orchestrator": {"renderer": {"name": "default"}},
            }
        )


def test_tokenizer_name_falls_back_to_model_name_when_unset():
    config = RLConfig.model_validate(
        {
            "model": {"name": "my-model"},
            "tokenizer": {"trust_remote_code": True},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.tokenizer.name == "my-model"
    assert config.orchestrator.tokenizer.name == "my-model"
    assert config.trainer.tokenizer.trust_remote_code is True
    assert config.orchestrator.tokenizer.trust_remote_code is True


def test_explicit_subconfig_tokenizer_name_survives_shared_model_propagation():
    """Regression: shared ``[model] name = "M"`` must propagate model names but
    must NOT clobber an explicit ``[orchestrator.tokenizer] name = "T"``.

    This is the case that the old RL-level ``auto_setup_tokenizer`` fix-up got
    wrong: it unconditionally re-derived ``orchestrator.tokenizer.name`` from
    ``orchestrator.model.name`` after propagation, silently overriding
    the user's explicit value. The ``mode="before"`` ``auto_setup_shared_configs``
    propagator fixes this because it propagates the model name into the raw
    dict before sub-configs are built, so ``OrchestratorConfig``'s own
    ``auto_setup_tokenizer`` (mode=after) sees the resolved name *and* the
    explicit user-set tokenizer name, and the ``fill``-if-absent semantic
    leaves the explicit value alone.
    """
    config = RLConfig.model_validate(
        {
            "model": {"name": "M"},
            "trainer": {},
            "orchestrator": {
                "renderer": {"name": "default"},
                "tokenizer": {"name": "explicit-orch-tok"},
            },
        }
    )
    # Shared model.name reached every sub-config that didn't override it.
    assert config.trainer.model.name == "M"
    assert config.orchestrator.model.name == "M"
    # Trainer didn't specify a tokenizer, so it falls back to the propagated model name.
    assert config.trainer.tokenizer.name == "M"
    # Orchestrator's explicit tokenizer name survived.
    assert config.orchestrator.tokenizer.name == "explicit-orch-tok"


def test_tokenizer_chat_template_mismatch_raises():
    with pytest.raises(ValidationError, match="chat_template"):
        RLConfig.model_validate(
            {
                "trainer": {"tokenizer": {"chat_template": "A"}},
                "orchestrator": {"renderer": {"name": "default"}, "tokenizer": {"chat_template": "B"}},
            }
        )


def test_shared_seq_len_propagates_to_subconfigs():
    config = RLConfig.model_validate(
        {
            "seq_len": 4096,
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.model.seq_len == 4096
    assert config.orchestrator.seq_len == 4096


def test_shared_and_sub_seq_len_conflict_raises():
    """Setting seq_len at the shared level and on a sub-config is a conflict —
    forces the user to pick one place to express the value rather than
    relying on the silent 'sub wins' rule."""
    with pytest.raises(ValidationError, match=r"seq_len.*trainer.model.seq_len"):
        RLConfig.model_validate(
            {
                "seq_len": 4096,
                "trainer": {"model": {"seq_len": 8192}},
                "orchestrator": {"renderer": {"name": "default"}},
            }
        )


def test_shared_and_sub_model_name_conflict_raises():
    """Setting model.name at the shared level and on a sub-config is a conflict."""
    with pytest.raises(ValidationError, match=r"model.name.*trainer.model.name"):
        RLConfig.model_validate(
            {
                "model": {"name": "X"},
                "trainer": {"model": {"name": "Y"}},
                "orchestrator": {"renderer": {"name": "default"}},
            }
        )


def test_shared_and_sub_max_steps_conflict_raises():
    """Top-level scalar shared fields also participate in the mutex check."""
    with pytest.raises(ValidationError, match=r"max_steps.*orchestrator.max_steps"):
        RLConfig.model_validate(
            {
                "max_steps": 100,
                "trainer": {},
                "orchestrator": {"renderer": {"name": "default"}, "max_steps": 200},
            }
        )


def test_trainer_chat_template_cascades_to_inference():
    """``[trainer.tokenizer] chat_template`` set directly (no shared
    ``[tokenizer] chat_template``) must still reach
    ``inference.model.chat_template`` so vLLM's ``--chat-template`` is wired
    up. Regression: the original ``auto_setup_tokenizer`` cascaded this; the
    refactored propagator must keep doing it."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "trainer": {"tokenizer": {"chat_template": "TPL"}},
            "orchestrator": {"renderer": {"name": "default"}, "tokenizer": {"chat_template": "TPL"}},
            "inference": {},
        }
    )
    assert config.trainer.tokenizer.chat_template == "TPL"
    assert config.orchestrator.tokenizer.chat_template == "TPL"
    assert config.inference is not None
    assert config.inference.model.chat_template == "TPL"


def test_shared_wandb_fields_propagate_to_subconfigs():
    """Every ``SharedWandbConfig`` leaf (project, entity, name, group, tags,
    offline) propagates to both trainer.wandb and orchestrator.wandb. Regression
    for a miss in the inline propagator."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "wandb": {
                "project": "shared-proj",
                "entity": "shared-entity",
                "name": "shared-name",
                "group": "shared-group",
                "tags": ["a", "b"],
                "offline": False,
            },
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    for component in (config.trainer.wandb, config.orchestrator.wandb):
        assert component is not None
        assert component.project == "shared-proj"
        assert component.entity == "shared-entity"
        assert component.name == "shared-name"
        assert component.group == "shared-group"
        assert component.tags == ["a", "b"]
        assert component.offline is False


def test_empty_shared_ckpt_block_does_not_conflict_with_subconfig_ckpt():
    """An empty shared [ckpt] block is a presence-only signal, not a field
    setting — it should not conflict with a non-empty [trainer.ckpt]."""
    config = RLConfig.model_validate(
        {
            "ckpt": {},  # empty block, no field set
            "trainer": {"ckpt": {"interval": 50}},
            "orchestrator": {"renderer": {"name": "default"}, "ckpt": {"interval": 50}},
        }
    )
    assert config.trainer.ckpt is not None
    assert config.trainer.ckpt.interval == 50


def test_shared_and_subconfig_disjoint_fields_coexist():
    """Per-field mutex only forbids conflicts on the SAME field — disjoint
    fields in [model] vs [trainer.model] are fine."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "trainer": {"model": {"impl": "custom"}},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.model.name == "Qwen/Qwen3-0.6B"
    assert config.trainer.model.impl == "custom"


def test_shared_output_dir_propagates_through_cli(tmp_path):
    """Shared output_dir from CLI reaches sub-configs even when tyro constructs sub-configs before the before-validator."""
    toml_path = tmp_path / "cfg.toml"
    write_toml(
        toml_path,
        {
            "max_steps": 1,
            "seq_len": 128,
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "trainer": {},
            "orchestrator": {"batch_size": 16, "group_size": 1},
            "inference": {},
        },
    )
    shared_out = tmp_path / "shared"
    config = cli(RLConfig, args=["@", str(toml_path), "--output-dir", str(shared_out)])
    assert config.trainer.output_dir == shared_out
    assert config.orchestrator.output_dir == shared_out / "run_default"


def test_orchestrator_renderer_auto_rejects_unmapped_model():
    """Default ``renderer`` (AutoRendererConfig) must reject models not in MODEL_RENDERER_MAP."""
    with pytest.raises(ValidationError, match="silently fall back to DefaultRenderer"):
        OrchestratorConfig.model_validate({"model": {"name": "not-a-real-org/not-a-real-model"}})


def test_orchestrator_renderer_auto_accepts_mapped_model():
    """The default Qwen model is in MODEL_RENDERER_MAP and should validate cleanly."""
    config = OrchestratorConfig.model_validate({"model": {"name": "Qwen/Qwen3-0.6B"}})
    assert config.renderer is not None
    assert config.renderer.name == "auto"


def test_orchestrator_explicit_renderer_skips_unmapped_check():
    """Explicit renderer.name bypasses the auto-resolution check — user opted in."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "renderer": {"name": "qwen3"},
        }
    )
    assert config.renderer is not None
    assert config.renderer.name == "qwen3"


def test_orchestrator_renderer_none_rejected():
    """A renderer is required (training is renderer-only): the non-optional type rejects None."""
    with pytest.raises(ValidationError, match="renderer"):
        OrchestratorConfig.model_validate(
            {
                "model": {"name": "not-a-real-org/not-a-real-model"},
                "renderer": None,
            }
        )


def test_orchestrator_explicit_default_renderer_with_unmapped_model():
    """renderer.name='default' is an explicit opt-in to DefaultRenderer and must pass."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "renderer": {"name": "default", "tool_parser": "qwen3"},
        }
    )
    assert config.renderer is not None
    assert config.renderer.name == "default"
    assert config.renderer.tool_parser == "qwen3"


def test_shared_model_name_resolves_inference_parsers():
    """Shared [model] name must reach inference.model BEFORE ModelConfig's after-validator
    runs auto_resolve_parsers — i.e. the parsers resolve from the propagated name, not
    from an empty default.
    """
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {},
        }
    )
    assert config.inference is not None
    assert config.inference.model.name == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert config.inference.model.tool_call_parser == "qwen3_coder"


def test_explicit_inference_parser_wins_over_auto():
    """Explicit inference.model.tool_call_parser is preserved even when the shared model
    name would otherwise auto-resolve to something else."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {"model": {"tool_call_parser": "hermes"}},
        }
    )
    assert config.inference is not None
    assert config.inference.model.tool_call_parser == "hermes"
