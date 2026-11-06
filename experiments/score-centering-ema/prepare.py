"""Generate reviewed configs/manifests only. This command never submits a job."""

import argparse
import copy
import json
from pathlib import Path

import tomli_w
import yaml
from pydantic import model_validator

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import TrainerConfig

MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL = f"/data/cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/{MODEL_REVISION}"
SOURCE = "/data/siddarth-sc-ema"
IMAGE = "primeintellect/prime-rl@sha256:e175fb968a270de9450cde72817ebac0a97087fbfc3b7765f1c10cc5b5886209"


class KubernetesConfig(RLConfig):
    @model_validator(mode="after")
    def validate_deployment(self):
        d = self.deployment
        if d.type != "multi_node" or self.slurm is not None:
            raise ValueError("External Kubernetes placement required")
        if (d.num_train_nodes, d.num_infer_nodes, d.num_infer_replicas) != (2, 1, 2):
            raise ValueError("Expected two trainer nodes and two independent inference nodes")
        if self.value_function is not None:
            raise ValueError("No value functions in this comparison")
        return self


def configuration(arm, seed, *, beta=0.01, top_k=128, steps=2000, lag=8):
    name = f"siddarth-rg-{arm}-s{seed}-k{top_k}-b{beta:g}".replace(".", "p")
    out = f"/data/outputs/{name}"
    sampling = {
        "temperature": 1.0,
        "extra_body": {
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
    }
    if arm == "sc":
        sampling["extra_body"]["extra_args"] = {"score_centering_top_k": top_k}
    wandb = {
        "entity": "swish",
        "project": "siddarth-prime-values",
        "group": name,
        "offline": False,
        "tags": ["score-centering-ema", arm, f"seed-{seed}"],
    }
    endpoints = [f"http://{name}-infer-{node}:{8000 + gpu}/v1" for node in range(2) for gpu in range(8)]
    data = {
        "output_dir": out,
        "model": {"name": MODEL},
        "max_steps": steps,
        "seq_len": 8192,
        "ckpt": {"interval": 100, "keep_last": 2},
        "trainer": {
            "seed": seed,
            "rollout_retention_steps": 2,
            "weight_broadcast": {"type": "filesystem", "acknowledged_cleanup": True},
            "model": {
                "impl": "custom",
                "attn": "flash_attention_2",
                "dp_replicate": 2,
                "optim_cpu_offload": False,
                "fused_lm_head_token_chunk_size": 1024,
            },
            "optim": {"lr": 1e-6},
            "loss": {
                "type": "research",
                "importance_cap": 2.0,
                "score_centering": arm == "sc",
                "top_k": top_k,
                "ema_kl_beta": beta if arm == "ema" else 0.0,
                "ema_decay": 0.95,
            },
            "wandb": {**wandb, "name": name + "-trainer"},
        },
        "inference": {
            "seed": seed,
            "parallel": {"tp": 1},
            "model": {"max_model_len": 8192},
            "vllm_extra": {"max_logprobs": top_k, "generation_config": "vllm"},
        },
        "orchestrator": {
            "renderer": {"name": "qwen3"},
            "checkpoint_weights_dir": out + "/weights",
            "seed": seed,
            "batch_size": 256,
            "post_batch_filters": [
                {"type": t, "enforce": False} for t in ("gibberish", "repetition", "zero_advantage")
            ],
            "group_size": 8,
            "max_inflight_rollouts": 512,
            "max_async_batches": lag,
            "max_off_policy_steps": 16,
            "model": {"client": {"base_url": endpoints, "admin_base_url": endpoints}},
            "algo": {"type": "grpo", "baseline": {"type": "mean"}},
            "train": {
                "sampling": sampling,
                "env": [
                    {
                        "id": "rg-mix-env",
                        "args": {
                            "dataset_path": "/data/rg-dataset",
                            "num_train_examples": 10000,
                            "num_eval_examples": 256,
                            "seed": 42,
                        },
                    }
                ],
            },
            "wandb": {**wandb, "name": name + "-orchestrator"},
        },
        "deployment": {
            "type": "multi_node",
            "gpus_per_node": 8,
            "num_train_nodes": 2,
            "num_infer_nodes": 1,
            "num_infer_replicas": 2,
        },
    }
    c = KubernetesConfig.model_validate(data)
    c.orchestrator.model.client.base_url = endpoints
    c.orchestrator.model.client.admin_base_url = endpoints
    assert c.trainer.model.optimization_dtype == c.trainer.model.reduce_dtype == "float32"
    assert c.orchestrator.group_size == 8 and c.orchestrator.batch_size == 256
    return name, c


def manifest(name, seed, config_dir):
    resources = {"cpu": "32", "memory": "256Gi", "nvidia.com/gpu": 8, "nvidia.com/hostdev": 8}
    env = {
        "UV_PROJECT_ENVIRONMENT": "/app/.venv",
        "UV_CACHE_DIR": "/data/cache/uv",
        "HF_HOME": "/data/cache/huggingface",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VERIFIERS": "0.1.15.dev411",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RENDERERS": "0.1.8.dev28",
        "WANDB_MODE": "online",
        "NCCL_DEBUG": "INFO",
        "VLLM_ENGINE_READY_TIMEOUT_S": "4200",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "RESEARCH_RUN_NAME": name,
        "RESEARCH_SEED": str(seed),
        "RESEARCH_CONFIG_DIR": str(config_dir),
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": "siddarth", "labels": {"kueue.x-k8s.io/queue-name": "gpu"}},
        "spec": {
            "suspend": True,
            "completionMode": "Indexed",
            "completions": 4,
            "parallelism": 4,
            "backoffLimit": 0,
            "activeDeadlineSeconds": 259200,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 60,
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "fsGroup": 1000,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                    "containers": [
                        {
                            "name": "run",
                            "image": IMAGE,
                            "workingDir": SOURCE,
                            "command": ["bash", "-lc"],
                            "args": [
                                "set -eu; uv sync --all-extras --frozen; exec uv run --no-sync python experiments/score-centering-ema/launch.py"
                            ],
                            "env": [{"name": k, "value": v} for k, v in env.items()]
                            + [
                                {
                                    "name": "WANDB_API_KEY",
                                    "valueFrom": {"secretKeyRef": {"name": "siddarth-wandb", "key": "api-key"}},
                                },
                                {
                                    "name": "JOB_COMPLETION_INDEX",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.labels['batch.kubernetes.io/job-completion-index']"
                                        }
                                    },
                                },
                            ],
                            "resources": {"requests": resources, "limits": resources},
                            "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                            "volumeMounts": [
                                {"name": "shared", "mountPath": "/data"},
                                {"name": "shm", "mountPath": "/dev/shm"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "shared", "persistentVolumeClaim": {"claimName": "prime-values-shared"}},
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"}},
                    ],
                }
            },
        },
    }
    services = []
    for i, suffix in enumerate(["infer-0", "infer-1", "trainer", "trainer-worker"]):
        ports = range(8000, 8008) if i < 2 else [29500]
        services.append(
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": f"{name}-{suffix}", "namespace": "siddarth"},
                "spec": {
                    "clusterIP": "None",
                    "publishNotReadyAddresses": True,
                    "selector": {
                        "batch.kubernetes.io/job-name": name,
                        "batch.kubernetes.io/job-completion-index": str(i),
                    },
                    "ports": [{"name": f"p{p}", "port": p, "targetPort": p} for p in ports],
                },
            }
        )
    return [*services, job]


def write(arm, seed, dest, **kwargs):
    name, c = configuration(arm, seed, **kwargs)
    folder = dest / name
    folder.mkdir(parents=True, exist_ok=True)
    for role, sub, schema in [
        ("trainer", c.trainer, TrainerConfig),
        ("orchestrator", c.orchestrator, OrchestratorConfig),
        ("inference", c.inference, InferenceConfig),
    ]:
        exclude = {"deployment", "slurm", "output_dir", "dry_run"} if role == "inference" else set()
        payload = sub.model_dump(mode="json", exclude_none=True, exclude=exclude)
        reloaded = schema.model_validate(copy.deepcopy(payload))
        if reloaded.model_dump(mode="json", exclude_none=True, exclude=exclude) != payload:
            raise ValueError(f"{role} changed during serialized-config roundtrip")
        (folder / f"{role}.toml").write_text(tomli_w.dumps(payload))
    (folder / "resolved.json").write_text(c.model_dump_json(indent=2))
    remote_config = Path(SOURCE) / "experiments/score-centering-ema/prepared" / name
    (folder / "run.yaml").write_text(yaml.safe_dump_all(manifest(name, seed, remote_config), sort_keys=False))
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "prepared")
    a = parser.parse_args()
    runs = [write(arm, seed, a.output) for arm in ("base", "sc", "ema") for seed in (11, 22, 33)]
    (a.output / "manifest.json").write_text(
        json.dumps(
            {
                "status": "PREPARED_NOT_LAUNCHED",
                "runs": runs,
                "model_revision": MODEL_REVISION,
                "simultaneous_run_limit": 3,
                "gpus_per_run": 32,
            },
            indent=2,
        )
    )
    print(f"Prepared {len(runs)} configs. No jobs submitted.")
