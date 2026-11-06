"""Authored launch contracts; no model download, worker allocation, or GPU execution."""

from pathlib import Path

import pytest
import verifiers.v1 as vf

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.entrypoints.rl import write_subconfigs
from prime_rl.utils.config import cli

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "configs/native").glob("*.toml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_native_launch_contract_and_actual_child_cli_roundtrip(path, tmp_path):
    config = cli(RLConfig, args=["@", str(path)])
    value_enabled = path.stem.endswith("_value")
    is_rg = path.stem.startswith("rg_mix")
    assert config.max_steps == 3
    assert config.seq_len == (2048 if is_rg else 512)
    assert config.orchestrator.group_size == config.orchestrator.batch_size == 8
    assert config.orchestrator.algo.baseline.type == ("value" if value_enabled else "mean")
    assert (config.value_function is not None) == value_enabled
    assert config.deployment.gpus_per_node == (4 if value_enabled else 2)
    assert config.deployment.num_train_gpus == config.deployment.num_infer_gpus == 1
    assert config.deployment.num_value_train_gpus == config.deployment.num_value_eval_gpus == int(value_enabled)
    assert config.trainer.model.optimization_dtype == config.trainer.model.reduce_dtype == "float32"
    assert config.trainer.wandb.offline and config.orchestrator.wandb.offline
    sampling = config.orchestrator.train.sampling.to_sampling_args()
    assert sampling["temperature"] == sampling["top_p"] == 1.0
    assert sampling["max_completion_tokens"] == (128 if is_rg else 64)
    assert sampling["extra_body"] == {
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    env = config.orchestrator.train.env[0]
    assert env.training_projection is None
    assert env.env.agent.runtime.type == "subprocess" and env.env.agent.harness.id == "null"
    taskset = vf.load_taskset(env.env.taskset)
    assert type(taskset).__name__ == ("RGMixTaskset" if is_rg else "ReverseTextTaskset")
    if is_rg:
        # Registry/factory construction must not require this host to mount the cache.
        assert env.env.taskset.dataset_path == Path("/data/rg-dataset")
        assert env.env.taskset.split == "train"
        assert env.env.taskset.num_train_examples == 10000
        assert env.env.taskset.num_eval_examples == 256
    if value_enabled:
        value = config.value_function
        assert value.warmup_updates == 2 and value.batch_size == 8
        assert value.replay.max_updates_per_rollout == 1
        assert value.evaluator.placement == "dedicated"
        assert value.model.optimization_dtype == value.model.reduce_dtype == "float32"

    write_subconfigs(config, tmp_path)
    children = [
        ("trainer", TrainerConfig, config.trainer),
        ("orchestrator", OrchestratorConfig, config.orchestrator),
        ("inference", InferenceConfig, config.inference),
    ]
    if value_enabled:
        children.append(("value", ValueFunctionConfig, config.value_function))
    for name, cls, expected in children:
        child = cli(cls, args=["@", str(tmp_path / f"{name}.toml")])
        exclude = {"deployment", "slurm", "output_dir", "dry_run"} if name == "inference" else set()
        assert child.model_dump(mode="json", exclude=exclude) == expected.model_dump(mode="json", exclude=exclude), name
