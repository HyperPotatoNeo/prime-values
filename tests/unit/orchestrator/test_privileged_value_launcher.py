from __future__ import annotations

import subprocess
import sys

import pytest

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints import rl as rl_entrypoint


def test_managed_launcher_propagates_privileged_overflow_and_stops_siblings(
    tmp_path,
    monkeypatch,
):
    """Run the real local-launcher monitor/cleanup boundary with controlled children."""
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {"algo": {"type": "grpo"}},
            "value_function": {"evaluator": {"placement": "trainer"}},
            "deployment": {"type": "single_node", "gpus_per_node": 3},
            "output_dir": tmp_path,
        }
    )
    overflow_program = """
from types import SimpleNamespace

from prime_rl.orchestrator.train_sink import TrainSink

sink = TrainSink.__new__(TrainSink)
sink._value_seq_len = 2
sink.renderer = SimpleNamespace(
    render_ids=lambda messages, add_generation_prompt=False: [1, 90]
)
sink.tokenizer = SimpleNamespace(bos_token_id=1)
rollout = SimpleNamespace(
    env_name="overflow-smoke",
    task=SimpleNamespace(idx=7, value_function_prompt="private oracle"),
)
sample = SimpleNamespace(token_ids=[1, 2])
sink._build_value_prefix(rollout, [sample], enabled=True)
"""
    real_popen = subprocess.Popen
    children: list[tuple[str, subprocess.Popen]] = []

    def controlled_popen(command, *args, **kwargs):
        if isinstance(command, list) and command[0] == "orchestrator":
            role = "orchestrator"
            program = overflow_program
        elif isinstance(command, list) and "--role=value-trainer" in command:
            role = "value-trainer"
            program = "import time; time.sleep(60)"
        elif isinstance(command, list) and "--role=trainer" in command:
            role = "trainer"
            program = "import time; time.sleep(60)"
        else:
            role = "tail"
            program = "import time; time.sleep(60)"
        process = real_popen(
            [sys.executable, "-c", program],
            env=kwargs.get("env"),
            stdout=kwargs.get("stdout"),
            stderr=kwargs.get("stderr"),
        )
        children.append((role, process))
        return process

    monkeypatch.setattr(rl_entrypoint, "Popen", controlled_popen)
    monkeypatch.setattr(rl_entrypoint, "get_physical_gpu_ids", lambda: [0, 1])
    monkeypatch.setattr(rl_entrypoint.signal, "signal", lambda *_args: None)
    # The production helper waits up to five seconds per live monitor before
    # killing children. Child termination, the behavior under test, remains the
    # real ``cleanup_processes`` call; skip only those pre-cleanup joins here.
    monkeypatch.setattr(rl_entrypoint, "cleanup_threads", lambda _threads: None)

    with pytest.raises(SystemExit) as exit_info:
        rl_entrypoint.rl_local(config)

    assert exit_info.value.code == 1
    assert {role for role, _ in children} == {
        "value-trainer",
        "orchestrator",
        "trainer",
        "tail",
    }
    assert all(process.poll() is not None for _, process in children)
    orchestrator_log = (tmp_path / "logs" / "orchestrator.log").read_text()
    assert "conditioned value input exceeds" in orchestrator_log
    assert "private oracle" not in orchestrator_log
