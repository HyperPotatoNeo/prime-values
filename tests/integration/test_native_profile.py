"""Bounded native lifecycle CI, not a reward-learning or model-parity benchmark.

The 0.6B model uses its explicit non-thinking prompt mode for the 64-token
answer budget. Four single-GPU roles target A6000 runners; hardware capacity
and lifecycle acceptance require actual execution, not inference
from earlier H200/4B receipts. Historical convergence tests remain unchanged.
"""

import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import write_subconfigs
from prime_rl.utils.config import cli

ROOT = Path(__file__).resolve().parents[2]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def native_config(root: Path, *, value: bool, resume_value: int | None = None) -> RLConfig:
    args = [
        "@",
        str(ROOT / f"configs/native/reverse_text_{'value' if value else 'policy'}.toml"),
        "--model.name",
        "Qwen/Qwen3-0.6B",
        "--no-orchestrator.renderer.enable-thinking",
        "--output-dir",
        str(root / "run"),
        "--weight-broadcast.type",
        "nccl",
        "--log.level",
        "debug",
        "--ckpt.keep-last",
        "3",
    ]
    if value:
        args += ["--value-function.ckpt.keep-last", "3"]
    config = cli(RLConfig, args=args)
    if resume_value is not None:
        # Re-parse the raw authored config, not its resolved hierarchy: shared
        # max_steps/checkpoint fields are intentionally propagated to children.
        import tomllib

        data = tomllib.loads((ROOT / f"configs/native/reverse_text_{'value' if value else 'policy'}.toml").read_text())
        data.update(
            max_steps=4, output_dir=str(root / "run"), weight_broadcast={"type": "nccl"}, log={"level": "debug"}
        )
        data["model"]["name"] = "Qwen/Qwen3-0.6B"
        data["orchestrator"]["renderer"]["enable_thinking"] = False
        data["ckpt"].update(resume_step=3, keep_last=3)
        if value:
            data["value_function"]["ckpt"].update(resume_step=resume_value, keep_last=3)
        config = RLConfig.model_validate(deepcopy(data))
    assert config.orchestrator.renderer.enable_thinking is False
    return config


def child_roundtrip(config: RLConfig, directory: Path) -> None:
    write_subconfigs(config, directory)
    for role, expected in (
        ("trainer", config.trainer),
        ("orchestrator", config.orchestrator),
        ("inference", config.inference),
        ("value", config.value_function),
    ):
        if expected is None:
            continue
        actual = cli(type(expected), args=["@", str(directory / f"{role}.toml")])
        exclude = {"deployment", "slurm", "output_dir", "dry_run"} if role == "inference" else set()
        assert actual.model_dump(mode="json", exclude=exclude) == expected.model_dump(mode="json", exclude=exclude)


def read_log(path: Path) -> str:
    return ANSI.sub("", path.read_text(errors="replace"))


def checkpoint_progress(config: RLConfig, step: int) -> dict:
    from torch.distributed.checkpoint import load

    directory = config.trainer.output_dir / "checkpoints" / f"step_{step}" / "trainer"
    assert (directory / ".metadata").is_file() and list(directory.glob("*.distcp"))
    state = {"app": {"progress": {"step": 0, "total_tokens": 0, "total_samples": 0}}}
    load(state, checkpoint_id=directory, no_dist=True)
    return state["app"]["progress"]


def prepare_stage(config: RLConfig, directory: Path) -> Path:
    directory.mkdir()
    child_roundtrip(config, directory / "expected-configs")
    # Preserve the authored request hierarchy so shared fields are resolved once.
    request = directory / "requested.json"
    import tomllib

    value = config.value_function is not None
    raw = tomllib.loads((ROOT / f"configs/native/reverse_text_{'value' if value else 'policy'}.toml").read_text())
    raw.update(
        max_steps=config.max_steps,
        output_dir=str(config.output_dir),
        weight_broadcast={"type": "nccl"},
        log={"level": "debug"},
    )
    raw["model"]["name"] = config.model.name
    raw["orchestrator"]["renderer"]["enable_thinking"] = False
    raw["ckpt"].update(keep_last=3)
    if config.ckpt.resume_step is not None:
        raw["ckpt"]["resume_step"] = config.ckpt.resume_step
    if value:
        raw["value_function"]["ckpt"].update(keep_last=3)
        if config.value_function.ckpt.resume_step is not None:
            raw["value_function"]["ckpt"]["resume_step"] = config.value_function.ckpt.resume_step
    assert RLConfig.model_validate(deepcopy(raw)) == config
    request.write_text(json.dumps(raw, indent=2) + "\n")
    (directory / "resolved.json").write_text(config.model_dump_json(indent=2) + "\n")
    return request


def stage(config: RLConfig, directory: Path) -> dict:
    request = prepare_stage(config, directory)
    value = config.value_function is not None
    logs = config.output_dir / "logs"
    if logs.exists():
        logs.rename(directory / "prelaunch-logs")
    rpc = directory / "vllm-ipc"
    rpc.mkdir()
    with (directory / "launcher.log").open("w") as output:
        process = subprocess.Popen(
            ["uv", "run", "--no-sync", "rl", "@", str(request)],
            cwd=ROOT,
            env={**os.environ, "WANDB_MODE": "offline", "VLLM_RPC_BASE_PATH": str(rpc)},
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=1500)
        finally:
            natural_returncode = process.poll()
            if natural_returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (directory / "exit.json").write_text(json.dumps({"natural_returncode": natural_returncode}) + "\n")
            if logs.exists():
                shutil.copytree(logs, directory / "logs")
    assert natural_returncode == 0, f"See {directory / 'launcher.log'}"
    spec = importlib.util.spec_from_file_location(
        "native_ci_log_checks", ROOT / "experiments/backend-sync/log_checks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launcher = read_log(directory / "launcher.log")
    for path in (directory / "logs").rglob("*.log"):
        module.classify_tracebacks(str(path.relative_to(directory / "logs")), read_log(path), launcher, final=True)
    trainer = read_log(directory / "logs/trainer.log")
    orchestrator = read_log(directory / "logs/orchestrator.log")
    assert "Training finished!" in launcher and "RL trainer finished!" in trainer
    assert "Orchestrator finished." in orchestrator and "Pipeline drained" in orchestrator
    rows = re.findall(r"(?<!Micro )Step (\d+) \|[^\n]*?Loss (\S+)", trainer)
    grads = re.findall(r"(?<!Micro )Step (\d+) \|[^\n]*?Grad\. Norm (\S+)", trainer)
    expected = [4] if config.ckpt.resume_step is not None else [1, 2, 3]
    assert [int(step) for step, _ in rows] == expected
    assert [step for step, _ in rows] == [step for step, _ in grads]
    assert all(math.isfinite(float(number)) for _, number in rows + grads)
    restored = re.search(r"Starting from step (\d+) \(total_tokens=(\d+), total_samples=(\d+)\)", trainer)
    assert restored
    initial = tuple(map(int, restored.groups()))
    snapshots = {step: checkpoint_progress(config, step) for step in expected}
    previous = checkpoint_progress(config, 3) if expected == [4] else {"total_tokens": 0, "total_samples": 0}
    assert initial == (expected[0], previous["total_tokens"], previous["total_samples"])
    expected_samples = previous["total_samples"]
    micro_batches = 0
    for line in trainer.splitlines():
        micro = re.search(r"Micro Step \d+/(\d+) \|", line)
        if micro:
            micro_batches = int(micro[1])
        completed = re.search(r"(?<!Micro )Step (\d+) \|", line)
        if completed:
            step = int(completed[1])
            assert micro_batches > 0, "Missing actual packed microbatch count"
            expected_samples += micro_batches
            progress = snapshots[step]
            assert progress["step"] == step
            assert progress["total_tokens"] > previous["total_tokens"]
            assert progress["total_samples"] == expected_samples
            previous = progress
            micro_batches = 0
    final = expected[-1]
    assert (config.orchestrator.output_dir / f"checkpoints/step_{final}/orchestrator/progress.pt").is_file()
    assert (config.output_dir / f"weights/step_{final}/STABLE").is_file()
    value_step = None
    if value:
        critic = read_log(directory / "logs/value_trainer.log")
        evaluator = read_log(directory / "logs/value_evaluator.log")
        updates = re.findall(r"Value version (\d+) \|[^\n]*?loss (\S+)", critic)
        assert updates and all(math.isfinite(float(loss)) for _, loss in updates)
        assert max(map(int, re.findall(r"Value evaluator adopted version (\d+)", evaluator)), default=-1) >= 2
        if expected == [4]:
            assert f"Resumed value trainer from version {config.value_function.ckpt.resume_step}" in critic
            assert all(int(version) > config.value_function.ckpt.resume_step for version, _ in updates)
        else:
            assert "Value warmup: gate version" in orchestrator
        versions = [
            int(p.name.removeprefix("step_"))
            for p in (config.output_dir / "value/checkpoints").glob("step_*")
            if (p / "STABLE").is_file()
        ]
        value_step = max(versions)
        assert value_step >= 2
        assert value_step == int(updates[-1][0]), "Final stable critic checkpoint must match its last actual update"
    else:
        assert not (directory / "logs/value_trainer.log").exists()
        assert not (directory / "logs/value_evaluator.log").exists()
    result = {
        "steps": expected,
        "progress": snapshots,
        "initial": initial,
        "value_checkpoint": value_step,
        "losses": rows,
        "gradient_norms": grads,
        "natural_returncode": natural_returncode,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_native(tmp_path: Path, *, value: bool) -> None:
    fresh_config = native_config(tmp_path, value=value)
    fresh = stage(fresh_config, tmp_path / "fresh")
    # A value-disabled run uses 0 only as the local resume selector; no critic
    # field is injected into its configuration.
    resumed = stage(
        native_config(tmp_path, value=value, resume_value=fresh["value_checkpoint"] or 0), tmp_path / "resumed"
    )
    assert resumed["initial"][1:] == (fresh["progress"][3]["total_tokens"], fresh["progress"][3]["total_samples"])


@pytest.mark.gpu
@pytest.mark.slow
def test_native_policy_fresh_resume(tmp_path):
    run_native(tmp_path, value=False)


@pytest.mark.gpu
@pytest.mark.slow
def test_native_value_fresh_resume(tmp_path):
    run_native(tmp_path, value=True)
