"""One real native-Episode policy update with the value plane disabled."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl-safety"))
from pipeline import (  # noqa: E402
    RLConfig,
    capture_checkpoint_progress,
    classify_tracebacks,
    cli,
    configuration,
    prepare_checkpoint_storage,
    stop,
    text,
    verify_environment,
    write_subconfigs,
)


def check_configs(config, directory, write):
    if write:
        write_subconfigs(config, directory)
    for role in ("trainer", "orchestrator", "inference"):
        expected = getattr(config, role)
        actual = cli(type(expected), args=["@", str(directory / f"{role}.toml")])
        exclude = {"deployment", "slurm", "output_dir", "dry_run"} if role == "inference" else set()
        assert actual.model_dump(exclude=exclude) == expected.model_dump(exclude=exclude), role
    assert not (directory / "value.toml").exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    prepare_checkpoint_storage(root, args.checkpoint_root)
    data, _ = configuration(root, "dedicated", args.model, 1)
    data["value_function"] = None
    data["orchestrator"]["algo"]["baseline"] = {"type": "mean"}
    data["deployment"].update(num_value_train_gpus=0, num_value_eval_gpus=0)
    config = RLConfig.model_validate(deepcopy(data))
    assert config.value_function is None and config.orchestrator.value_function is None
    assert config.trainer.model.optimization_dtype == config.trainer.model.reduce_dtype == "float32"
    evidence = root / "evidence"
    evidence.mkdir()
    request = evidence / "requested.json"
    request.write_text(json.dumps(data, indent=2) + "\n")
    (evidence / "resolved.json").write_text(config.model_dump_json(indent=2) + "\n")
    check_configs(config, evidence / "launch-configs", write=True)
    verify_environment(config, evidence)
    if args.prepare_only:
        return
    (root / "vllm-ipc").mkdir()
    snapshots = {}
    started = time.monotonic()
    with (evidence / "launcher.log").open("w") as log:
        process = subprocess.Popen(
            ["uv", "run", "--no-sync", "rl", "@", str(request)],
            env={**os.environ, "WANDB_MODE": "offline", "VLLM_RPC_BASE_PATH": str(root / "vllm-ipc")},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError("Value-disabled policy smoke exceeded its deadline")
                for path in (root / "run" / "logs").rglob("*.log"):
                    classify_tracebacks(
                        str(path.relative_to(root / "run" / "logs")),
                        text(path),
                        text(evidence / "launcher.log"),
                        final=False,
                    )
                capture_checkpoint_progress(root, evidence, snapshots)
                time.sleep(2)
            assert process.returncode == 0, process.returncode
            capture_checkpoint_progress(root, evidence, snapshots)
        finally:
            natural_exit = process.poll()
            stop(process)
            (evidence / "stage-exit.json").write_text(json.dumps({"natural_returncode": natural_exit}) + "\n")
            if (root / "run" / "logs").exists():
                shutil.copytree(root / "run" / "logs", evidence / "logs")
    check_configs(config, root / "run" / "configs", write=False)
    classifications = {}
    for path in (evidence / "logs").rglob("*.log"):
        name = str(path.relative_to(evidence / "logs"))
        classifications[name] = classify_tracebacks(name, text(path), text(evidence / "launcher.log"), final=True)
    trainer = text(evidence / "logs" / "trainer.log")
    rows = re.findall(r"Step (\d+) \|[^\n]*?Loss (\S+)", trainer)
    gradients = re.findall(r"Step (\d+) \|[^\n]*?Grad\. Norm (\S+)", trainer)
    assert [step for step, _ in rows] == [step for step, _ in gradients] == ["1"]
    assert all(math.isfinite(float(value)) for _, value in rows + gradients)
    assert "RL trainer finished!" in trainer
    assert "Orchestrator finished." in text(evidence / "logs" / "orchestrator.log")
    assert "Training finished!" in text(evidence / "launcher.log")
    assert not list((evidence / "logs").glob("value*.log"))
    assert not list((root / "run" / "value" / "checkpoints").glob("step_*"))
    assert snapshots["1"]["step"] == 1 and snapshots["1"]["total_tokens"] > 0
    micro = re.findall(r"Micro Step \d+/(\d+) \|", trainer)
    assert micro and snapshots["1"]["total_samples"] == int(micro[-1])
    assert (root / "run" / "weights" / "step_1" / "STABLE").exists()
    result = {
        "value_enabled": False,
        "policy_steps": [1],
        "policy_loss": float(rows[0][1]),
        "gradient_norm": float(gradients[0][1]),
        "natural_returncode": natural_exit,
        "checkpoint_progress": snapshots,
        "log_classifications": classifications,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
