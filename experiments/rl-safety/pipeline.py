"""Bounded real policy/critic pipeline and fresh-process restart validation.

Run serially in an isolated GPU pod. Pytest's process-cleanup fixture must not
run concurrently. This is a lifecycle check, not a reward benchmark.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path

import msgspec
from log_checks import classify_tracebacks

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import write_subconfigs
from prime_rl.utils.config import cli
from prime_rl.value.types import ValueVersionResponse

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def configuration(root, placement, model, steps, resume_step=None, value_resume_step=None):
    value = {
        "warmup_updates": 2,
        "batch_size": 8,
        "evaluator": {"placement": placement, "max_batch_tokens": 4096},
        "replay": {"max_updates_per_rollout": 1},
        "ckpt": {"interval": 1, "keep_last": 1},
    }
    if value_resume_step is not None:
        value["ckpt"]["resume_step"] = value_resume_step
    data = {
        "output_dir": str(root / "run"),
        "model": {"name": model},
        "seq_len": 512,
        "max_steps": steps,
        "log": {"level": "debug"},
        "ckpt": {"interval": 1, "keep_last": 1},
        "trainer": {
            "model": {
                "impl": "custom",
                "attn": "flash_attention_2",
                "optim_cpu_offload": False,
            },
            "optim": {"lr": 1e-6},
        },
        "inference": {
            "model": {"max_model_len": 512, "enforce_eager": True},
            "gpu_memory_utilization": 0.4,
            "enable_prefix_caching": False,
        },
        "orchestrator": {
            "batch_size": 8,
            "group_size": 2,
            "max_inflight_rollouts": 8,
            "algo": {"type": "grpo", "baseline": {"type": "value"}},
            "train": {
                "sampling": {"temperature": 1.0, "max_completion_tokens": 64},
                "env": [
                    {
                        "taskset": {"id": "reverse-text-v1"},
                        "harness": {"id": "null", "runtime": {"type": "subprocess"}},
                        "pool": {"type": "static", "num_workers": 1},
                    }
                ],
            },
        },
        "value_function": value,
        "deployment": {
            "type": "single_node",
            "gpus_per_node": 4,
            "num_infer_gpus": 1,
            "num_train_gpus": 1,
            "num_value_train_gpus": 1,
            "num_value_eval_gpus": 1 if placement == "dedicated" else 0,
        },
    }
    if resume_step is not None:
        data["ckpt"]["resume_step"] = resume_step
    config = RLConfig.model_validate(deepcopy(data))
    roundtrip = RLConfig.model_validate_json(config.model_dump_json())
    assert not roundtrip.clean_output_dir
    assert roundtrip.trainer.model.optimization_dtype == "float32"
    assert roundtrip.trainer.model.reduce_dtype == "float32"
    assert roundtrip.value_function.model.optimization_dtype == "float32"
    assert roundtrip.value_function.model.reduce_dtype == "float32"
    assert roundtrip.value_function.warmup_updates == 2
    assert roundtrip.orchestrator.batch_size == roundtrip.value_function.batch_size == 8
    sampling = roundtrip.orchestrator.train.sampling.to_sampling_args()
    assert sampling["temperature"] == sampling["top_p"] == 1.0
    assert sampling["max_completion_tokens"] == 64
    return data, roundtrip


def verify_launch_configs(config, directory, write=True):
    if write:
        write_subconfigs(config, directory)
    for role, expected in (
        ("trainer", config.trainer),
        ("orchestrator", config.orchestrator),
        ("inference", config.inference),
        ("value", config.value_function),
    ):
        actual = cli(type(expected), args=["@", str(directory / f"{role}.toml")])
        (directory / f"{role}.reloaded.json").write_text(actual.model_dump_json(indent=2) + "\n")
        exclude = {"deployment", "slurm", "output_dir", "dry_run"} if role == "inference" else set()
        before = expected.model_dump(mode="json", exclude=exclude)
        after = actual.model_dump(mode="json", exclude=exclude)
        changed = [key for key in before if before[key] != after[key]]
        assert not changed, f"{role} changed after launcher serialization: {changed}; inspect {directory}"
        if role in ("trainer", "value"):
            assert actual.model.optimization_dtype == actual.model.reduce_dtype == "float32"
            assert actual.model.compile is not None and actual.model.ac is not None


def prepare_checkpoint_storage(root, checkpoint_root):
    if checkpoint_root.exists() or checkpoint_root.is_symlink():
        raise FileExistsError(f"Use a fresh local checkpoint path: {checkpoint_root}")
    checkpoint_root = checkpoint_root.resolve()
    if root.is_relative_to(checkpoint_root) or checkpoint_root.is_relative_to(root):
        raise ValueError("Local checkpoint storage must be separate from the persistent evidence directory")
    checkpoint_root.mkdir(parents=True)
    links = {
        root / "run" / "checkpoints": checkpoint_root / "policy-checkpoints",
        root / "run" / "weights": checkpoint_root / "policy-weights",
        root / "run" / "value" / "checkpoints": checkpoint_root / "critic-checkpoints",
    }
    for link, target in links.items():
        target.mkdir()
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    (root / "checkpoint-storage.json").write_text(
        json.dumps(
            {
                "ephemeral": True,
                "resume_requires_same_node_storage": True,
                "links": {str(link.relative_to(root)): str(target) for link, target in links.items()},
            },
            indent=2,
        )
        + "\n"
    )


def verify_environment(config, evidence):
    from verifiers.v1.serve import EnvServer

    env_config = config.orchestrator.train.env[0]
    assert not env_config.is_legacy
    assert env_config.harness.id == "null"
    assert env_config.harness.runtime.type == "subprocess"
    server = EnvServer(env_config, address="tcp://127.0.0.1:0")
    try:
        assert len(server.tasks) >= config.orchestrator.batch_size
        assert server.tasks[0].answer == server.tasks[0].prompt[::-1]
        result = {
            "taskset": env_config.taskset.id,
            "taskset_class": type(server.env.taskset).__name__,
            "harness_class": type(server.env.harness).__name__,
            "num_tasks": len(server.tasks),
            "native_server_initialized": True,
        }
        (evidence / "environment-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        server.frontend.close()
        server.ctx.term()


def text(path):
    return ANSI.sub("", path.read_text(errors="replace")) if path.exists() else ""


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def capture_checkpoint_progress(root, evidence, snapshots):
    from torch.distributed.checkpoint import load

    for metadata in sorted((root / "run" / "checkpoints").glob("step_*/trainer/.metadata")):
        step = int(metadata.parent.parent.name.removeprefix("step_"))
        if str(step) in snapshots:
            continue
        state = {"app": {"progress": {"step": 0, "total_tokens": 0, "total_samples": 0}}}
        load(state, checkpoint_id=metadata.parent, no_dist=True)
        snapshots[str(step)] = state["app"]["progress"]
        (evidence / "checkpoint-progress.json").write_text(json.dumps(snapshots, indent=2) + "\n")


def run_stage(root, stage, data, config, timeout):
    evidence = root / stage
    evidence.mkdir()
    request = evidence / "requested.json"
    request.write_text(json.dumps(data, indent=2) + "\n")
    (evidence / "resolved.json").write_text(config.model_dump_json(indent=2) + "\n")
    verify_launch_configs(config, evidence / "launch-configs")
    verify_environment(config, evidence)
    rpc = root / "vllm-ipc"
    rpc.mkdir(exist_ok=True)
    env = {**os.environ, "WANDB_MODE": "offline", "VLLM_RPC_BASE_PATH": str(rpc)}
    live_logs = root / "run" / "logs"
    if live_logs.exists():
        live_logs.rename(evidence / "prelaunch-logs")
    started = time.monotonic()
    versions = []
    snapshots = {}
    with (evidence / "launcher.log").open("w") as log:
        process = subprocess.Popen(
            ["uv", "run", "--no-sync", "rl", "@", str(request)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"{stage} exceeded {timeout}s")
                for path in (root / "run" / "logs").rglob("*.log"):
                    classify_tracebacks(
                        str(path.relative_to(root / "run" / "logs")),
                        text(path),
                        text(evidence / "launcher.log"),
                        final=False,
                    )
                capture_checkpoint_progress(root, evidence, snapshots)
                for endpoint in config.value_function.evaluator.base_url:
                    try:
                        with urllib.request.urlopen(endpoint.rstrip("/") + "/version", timeout=1) as response:
                            version = msgspec.msgpack.decode(response.read(), type=ValueVersionResponse).version
                        versions.append(
                            {"elapsed": time.monotonic() - started, "endpoint": endpoint, "version": version}
                        )
                    except (urllib.error.URLError, TimeoutError):
                        # The endpoint is unavailable while its serving model starts.
                        pass
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f"{stage} exited {process.returncode}; see {evidence / 'launcher.log'}")
            capture_checkpoint_progress(root, evidence, snapshots)
        finally:
            natural_returncode = process.poll()
            stop(process)
            (evidence / "stage-exit.json").write_text(
                json.dumps(
                    {"natural_returncode": natural_returncode, "returncode_after_cleanup": process.returncode},
                    indent=2,
                )
                + "\n"
            )
            (evidence / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
            logs = root / "run" / "logs"
            if logs.exists():
                shutil.copytree(logs, evidence / "logs")
            configs = root / "run" / "configs"
            if configs.exists():
                shutil.copytree(configs, evidence / "actual-launch-configs")
    verify_launch_configs(config, evidence / "actual-launch-configs", write=False)
    return evidence, versions


def verify(
    root, evidence, versions, placement, final_step, resume_step=None, value_resume_step=None, allow_missing_exit=False
):
    log_classifications = {}
    for path in (evidence / "logs").rglob("*.log"):
        relative_path = str(path.relative_to(evidence / "logs"))
        log_classifications[relative_path] = classify_tracebacks(
            relative_path, text(path), text(evidence / "launcher.log"), final=True
        )
    exit_path = evidence / "stage-exit.json"
    if exit_path.exists():
        exit_evidence = json.loads(exit_path.read_text())
        assert exit_evidence["natural_returncode"] == 0, exit_evidence
    else:
        assert allow_missing_exit and resume_step is None, "Missing natural process-exit evidence"
        exit_evidence = {
            "natural_returncode": None,
            "qualification": "Historical fresh-stage exit was not recorded; completion logs and checkpoints verified.",
        }
    trainer = text(evidence / "logs" / "trainer.log")
    orchestrator = text(evidence / "logs" / "orchestrator.log")
    critic = text(evidence / "logs" / "value_trainer.log")
    assert "RL trainer finished!" in trainer
    assert "Orchestrator finished." in orchestrator
    assert "Training finished!" in text(evidence / "launcher.log")
    policy_rows = re.findall(r"Step (\d+) \|[^\n]*?Loss (\S+)", trainer)
    gradient_rows = re.findall(r"Step (\d+) \|[^\n]*?Grad\. Norm (\S+)", trainer)
    value_rows = re.findall(r"Value version (\d+) \|[^\n]*?loss (\S+)", critic)
    assert policy_rows, "No actual policy optimizer update was logged"
    assert max(int(step) for step, _ in policy_rows) == final_step, policy_rows
    assert value_rows, "No actual critic optimizer update was logged"
    assert all(math.isfinite(float(loss)) for _, loss in policy_rows + value_rows)
    assert [step for step, _ in gradient_rows] == [step for step, _ in policy_rows], "Missing actor gradient norm"
    assert all(math.isfinite(float(norm)) for _, norm in gradient_rows), gradient_rows
    snapshots = json.loads((evidence / "checkpoint-progress.json").read_text())
    if resume_step is None:
        assert "1" in snapshots, "No in-loop step-1 checkpoint captured; final-only checks miss counter lag"
    start = re.search(r"Starting from step (\d+) \(total_tokens=(\d+), total_samples=(\d+)\)", trainer)
    assert start, "Missing trainer restored progress counters"
    previous_tokens = int(start[2])
    expected_samples = int(start[3])
    if resume_step is None:
        assert (int(start[1]), previous_tokens, expected_samples) == (1, 0, 0)
    else:
        restored = snapshots[str(resume_step)]
        assert int(start[1]) == resume_step + 1
        assert previous_tokens == restored["total_tokens"]
        assert expected_samples == restored["total_samples"]
    micro_batches = 0
    for line in trainer.splitlines():
        micro = re.search(r"Micro Step \d+/(\d+) \|", line)
        if micro:
            micro_batches = int(micro[1])
        completed = re.search(r"(?<!Micro )Step (\d+) \|", line)
        if completed:
            assert micro_batches > 0, "Missing actual packed microbatch count"
            expected_samples += micro_batches
            step = completed[1]
            if step in snapshots:
                progress = snapshots[step]
                assert progress["step"] == int(step)
                assert progress["total_samples"] == expected_samples, (step, progress, expected_samples)
                assert progress["total_tokens"] > previous_tokens, (step, progress, previous_tokens)
                previous_tokens = progress["total_tokens"]
            micro_batches = 0
    assert str(final_step) in snapshots
    observed = [row["version"] for row in versions]
    assert observed and max(observed) >= 2, "Never observed a live evaluator at warmup version"
    if placement == "dedicated":
        adoptions = list(
            map(
                int,
                re.findall(r"Value evaluator adopted version (\d+)", text(evidence / "logs" / "value_evaluator.log")),
            )
        )
        assert max(adoptions, default=-1) >= 2
    else:
        adoptions = []
    if resume_step is None:
        assert "Value warmup: gate version" in orchestrator
    else:
        assert f"Resuming training from checkpoint step {resume_step}" in trainer
        assert f"Resuming orchestrator from checkpoint step {resume_step}" in orchestrator
        assert f"Resumed value trainer from version {value_resume_step}" in critic
        assert min(int(step) for step, _ in policy_rows) > resume_step
    ckpt = root / "run" / "checkpoints" / f"step_{final_step}"
    # Full policy checkpoints publish DCP metadata; only exported weights and
    # critic checkpoints have STABLE markers.
    assert (ckpt / "trainer" / ".metadata").exists()
    assert list((ckpt / "trainer").glob("*.distcp"))
    resolved = json.loads((evidence / "resolved.json").read_text())
    orchestrator_ckpt = Path(resolved["orchestrator"]["output_dir"]) / "checkpoints" / f"step_{final_step}"
    assert (orchestrator_ckpt / "orchestrator" / "progress.pt").exists()
    weights = root / "run" / "weights" / f"step_{final_step}"
    assert (weights / "STABLE").exists()
    value_steps = sorted(
        int(path.name.removeprefix("step_"))
        for path in (root / "run" / "value" / "checkpoints").glob("step_*")
        if (path / "STABLE").exists()
    )
    assert value_steps and max(value_steps) >= 2
    manifest = []
    for directory in (ckpt, orchestrator_ckpt, weights, root / "run" / "value" / "checkpoints"):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                record = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
                if path.name in {".metadata", "progress.pt", "STABLE"}:
                    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest.append(record)
    (evidence / "checkpoint-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    orchestrator_prefix = str(orchestrator_ckpt.relative_to(root)) + "/"
    (evidence / "orchestrator-checkpoint-manifest.json").write_text(
        json.dumps([record for record in manifest if record["path"].startswith(orchestrator_prefix)], indent=2) + "\n"
    )
    result = {
        "placement": placement,
        "policy_steps": [int(step) for step, _ in policy_rows],
        "policy_losses": [float(loss) for _, loss in policy_rows],
        "policy_gradient_norms": [float(norm) for _, norm in gradient_rows],
        "value_versions": [int(step) for step, _ in value_rows],
        "value_losses": [float(loss) for _, loss in value_rows],
        "max_observed_evaluator_version": max(observed),
        "dedicated_adoptions": adoptions,
        "policy_checkpoint": final_step,
        "stable_value_checkpoints": value_steps,
        "resume_policy_step": resume_step,
        "resume_value_step": value_resume_step,
        "checkpoint_progress": snapshots,
        "stage_exit": exit_evidence,
        "log_classifications": log_classifications,
    }
    (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-root", type=Path, help="Fresh node-local checkpoint directory; evidence stays in --output"
    )
    parser.add_argument("--placement", choices=("dedicated", "trainer"), default="dedicated")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-from-fresh", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.continue_from_fresh:
        if args.prepare_only or args.checkpoint_root is not None:
            parser.error("--continue-from-fresh uses existing checkpoints and cannot combine with preparation options")
        if not root.is_dir() or (root / "resumed").exists():
            raise ValueError("Continuation requires an existing fresh-stage output with no resumed stage")
        evidence = root / "fresh"
        data = json.loads((evidence / "requested.json").read_text())
        config = RLConfig.model_validate(deepcopy(data))
        assert config.output_dir == root / "run" and config.max_steps == 3
        assert config.ckpt.resume_step is None
        assert json.loads(config.model_dump_json()) == json.loads((evidence / "resolved.json").read_text())
        verify_launch_configs(config, evidence / "actual-launch-configs", write=False)
        versions = json.loads((evidence / "versions.json").read_text())
    else:
        if root.exists():
            raise FileExistsError(f"Use a fresh output path: {root}")
        root.mkdir(parents=True)
        if args.checkpoint_root is not None:
            prepare_checkpoint_storage(root, args.checkpoint_root)
        data, config = configuration(root, args.placement, args.model, 3)
        if args.prepare_only:
            (root / "requested.json").write_text(json.dumps(data, indent=2) + "\n")
            (root / "resolved.json").write_text(config.model_dump_json(indent=2) + "\n")
            verify_launch_configs(config, root / "launch-configs")
            verify_environment(config, root)
            return
        evidence, versions = run_stage(root, "fresh", data, config, args.timeout)
    placement = config.value_function.evaluator.placement
    fresh = verify(root, evidence, versions, placement, 3, allow_missing_exit=args.continue_from_fresh)
    value_step = max(fresh["stable_value_checkpoints"])
    data = deepcopy(data)
    data["max_steps"] = 4
    data["ckpt"]["resume_step"] = 3
    data["value_function"]["ckpt"]["resume_step"] = value_step
    config = RLConfig.model_validate(deepcopy(data))
    evidence, versions = run_stage(root, "resumed", data, config, args.timeout)
    resumed = verify(root, evidence, versions, placement, 4, 3, value_step)
    assert resumed["checkpoint_progress"]["3"] == fresh["checkpoint_progress"]["3"]
    (root / "summary.json").write_text(json.dumps({"fresh": fresh, "resumed": resumed}, indent=2) + "\n")
    print(json.dumps({"fresh": fresh, "resumed": resumed}, indent=2))


if __name__ == "__main__":
    main()
