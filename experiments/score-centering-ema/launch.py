"""Supervise the four Kubernetes roles and release all roles on failure/completion."""

import os
import re
import signal
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
run_name = os.environ["RESEARCH_RUN_NAME"]
seed = int(os.environ["RESEARCH_SEED"])
output = Path("/data/outputs") / run_name
config = Path(os.environ["RESEARCH_CONFIG_DIR"])
role = int(os.environ["JOB_COMPLETION_INDEX"])
logs = output / "logs"
logs.mkdir(parents=True, exist_ok=True)
os.environ["PRIME_RL_RUN_DONE_FILE"] = str(output / "policy.done")
children = {}
handles = []


def config_path(component):
    path = config / f"{component}.json"
    return str(path if path.exists() else config / f"{component}.toml")


def spawn(name, args, extra=None):
    env = dict(os.environ)
    env.update(extra or {})
    env["PRIME_RL_METRICS_FILE"] = str(logs / f"{name}-node{role}-metrics.jsonl")
    f = (logs / f"{name}-node{role}.log").open("w", buffering=1)
    handles.append(f)
    children[name] = subprocess.Popen(
        ["uv", "run", "--no-sync", *args], cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
    )
    print(f"Started {name}: {children[name].pid}", flush=True)


def terminate():
    for p in children.values():
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while any(p.poll() is None for p in children.values()) and time.monotonic() < deadline:
        time.sleep(1)
    for p in children.values():
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)


def interrupted(signum, frame):
    raise SystemExit(f"Received signal {signum}")


signal.signal(signal.SIGTERM, interrupted)

try:
    if role < 2:
        for gpu in range(8):
            Path(f"/tmp/vllm-{gpu}").mkdir(exist_ok=True)
            spawn(
                f"inference-{role}-{gpu}",
                [
                    "inference",
                    "@",
                    config_path("inference"),
                    "--seed",
                    str(seed * 100 + role * 8 + gpu),
                    "--server.host",
                    "0.0.0.0",
                    "--server.port",
                    str(8000 + gpu),
                    "--parallel.dp",
                    "1",
                    "--data-parallel-size-local",
                    "1",
                    "--api-server-count",
                    "1",
                ],
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "VLLM_RPC_BASE_PATH": f"/tmp/vllm-{gpu}",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                },
            )
    elif role in (2, 3):
        rank = role - 2
        spawn(
            "trainer",
            [
                "torchrun",
                "--nnodes=2",
                "--nproc-per-node=8",
                f"--node-rank={rank}",
                f"--master-addr={run_name}-trainer",
                "--master-port=29500",
                "-m",
                "prime_rl.trainer.rl.train",
                "@",
                config_path("trainer"),
            ],
        )
        if rank == 0:
            spawn("orchestrator", ["orchestrator", "@", config_path("orchestrator")])
    else:
        raise ValueError(role)
    finished = set()
    while True:
        if (output / "failure.txt").exists():
            raise RuntimeError((output / "failure.txt").read_text())
        for name, child in children.items():
            logfile = logs / f"{name}-node{role}.log"
            with logfile.open("rb") as f:
                f.seek(max(0, logfile.stat().st_size - 20000))
                tail = re.sub(r"\x1b\[[0-9;]*m", "", f.read().decode(errors="replace"))
            if re.search(r"ERROR\s+Fatal error in (train_value|train)\b", tail):
                raise RuntimeError(f"{name} logged a fatal training error; inspect {logfile}")
            rc = child.poll()
            if rc is None or name in finished:
                continue
            if rc != 0:
                raise RuntimeError(f"{name} exited {rc}; inspect {logs / name}.log")
            if name not in ("trainer", "orchestrator"):
                raise RuntimeError(f"{name} exited unexpectedly")
            finished.add(name)
            (output / f"{name}-{role}.done").touch()
        if all((output / name).exists() for name in ["trainer-2.done", "trainer-3.done", "orchestrator-2.done"]):
            (output / "policy.done").touch()
        if (output / "policy.done").exists():
            break
        time.sleep(3)
except BaseException as exc:
    (output / "failure.txt").write_text(f"role {role}: {exc}\n")
    raise
finally:
    terminate()
    for f in handles:
        f.close()
