"""Real stateless-worker and local Harbor artifact checks; no model or GPU."""

import argparse
import asyncio
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.loaders import resolve_env_config
from verifiers.v1.serve import EnvServer
from verifiers.v1.tasksets.harbor.taskset import HarborConfig, HarborTask, parse_task
from verifiers.v1.utils.compile import resolve_runtime_config

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.orchestrator.envs import Env


def worker(config_data, payload, harbor_payload, result):
    server = harbor_server = None
    try:
        config = resolve_env_config(config_data)
        server = EnvServer(config, address="tcp://127.0.0.1:0")
        rebuilt = server._build_task(payload)
        assert server.num_tasks is None
        harbor_server = EnvServer(resolve_env_config({"taskset": {"id": "harbor"}}), address="tcp://127.0.0.1:0")
        harbor_task = harbor_server._build_task(harbor_payload)
        artifact = Path(harbor_task.data.task_dir) / "tests" / "test.sh"
        result.put(
            {
                "pid": os.getpid(),
                "task_data": rebuilt.data.model_dump(mode="json"),
                "artifact_sha256": hashlib.sha256(Path(artifact).read_bytes()).hexdigest(),
            }
        )
    finally:
        if server is not None:
            server.frontend.close()
            server.ctx.term()
        if harbor_server is not None:
            harbor_server.frontend.close()
            harbor_server.ctx.term()


def check(output):
    output.mkdir(parents=True, exist_ok=False)
    log = output / "dataset-loads.jsonl"
    config = resolve_env_config({"taskset": {"id": "task-ownership-fixture", "load_log": str(log.resolve())}})
    task = next(iter(vf.load_taskset(config.taskset).load()))
    payload = task.data.model_dump(mode="json")
    task.data.nested["x"].append("changed after selection")
    assert payload["nested"]["x"] == ["first"]

    artifact = output / "harbor" / "tests" / "test.sh"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("#!/bin/sh\nprintf 'fixture grading bytes\\n'\n")
    root = artifact.parent.parent
    (root / "instruction.md").write_text("Preserve this exact question")
    (root / "task.toml").write_text(
        '[environment]\ndocker_image = "fixture:image"\n[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["example.org"]\n'
    )
    harbor_data = parse_task(root, 7, HarborConfig())
    assert harbor_data.network_allow == ["example.org"]
    assert harbor_data.task_dir == str(root)
    assert harbor_data.prompt == "Preserve this exact question"
    harbor_task = HarborTask(harbor_data, vf.TaskConfig())
    assert resolve_runtime_config(vf.DockerConfig(), harbor_task).allow == ["example.org"]
    try:
        resolve_runtime_config(vf.PrimeConfig(), harbor_task)
    except ValueError as error:
        assert "network policy" in str(error)
    else:
        raise AssertionError("Restricted Harbor task silently admitted on unsupported runtime")
    local = Env(EnvConfig.model_validate({"env": {"taskset": {"id": "harbor"}}}))
    local._task_class = HarborTask
    assert local.validate_task(harbor_task) is harbor_task
    try:
        local.validate_task(
            HarborTask(harbor_data.model_copy(update={"task_dir": str(output / "absent")}), vf.TaskConfig())
        )
    except ValueError as error:
        assert "grading artifacts are missing" in str(error)
    else:
        raise AssertionError("Missing grading artifacts admitted")
    remote = Env(EnvConfig.model_validate({"address": "tcp://127.0.0.1:1", "env": {"taskset": {"id": "harbor"}}}))
    try:
        asyncio.run(remote.start(output / "unused-logs"))
    except ValueError as error:
        assert "verified shared grading-artifact paths" in str(error)
        assert remote._env_client is None and remote._env_server_process is None
    else:
        raise AssertionError("Unverified external Harbor worker admitted")

    ctx = mp.get_context("spawn")
    result = ctx.Queue()
    processes = [
        ctx.Process(
            target=worker, args=(config.model_dump(mode="json"), payload, harbor_data.model_dump(mode="json"), result)
        )
        for _ in range(2)
    ]
    records = []
    try:
        for process in processes:
            process.start()
        records = [result.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0, f"worker {process.pid} did not exit successfully"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result.close()
        result.join_thread()
    loads = [json.loads(line) for line in log.read_text().splitlines()]
    assert loads == [{"pid": os.getpid(), "event": "load"}]
    assert len({record["pid"] for record in records}) == 2
    assert all(record["task_data"] == payload for record in records)
    assert all(record["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest() for record in records)
    receipt = {
        "verifier_module": vf.__file__,
        "client_load_count": len(loads),
        "worker_load_count": 0,
        "workers": records,
        "harbor": {
            "network_allow": harbor_data.network_allow,
            "task_dir": harbor_data.task_dir,
            "shared_local_artifact": True,
            "local_admitted": True,
            "missing_artifacts_rejected": True,
            "external_worker_rejected_before_io": True,
            "docker_policy_resolved": True,
            "unsupported_prime_policy_rejected": True,
        },
        "limits": "Actual EnvServer reconstruction in two local processes; actual Harbor 0.14.0 parser. No remote filesystem, Docker egress enforcement, or model-training claim. Separate producer smoke covers ZMQ and harness execution.",
    }
    (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    check(parser.parse_args().output.resolve())
