from types import SimpleNamespace

from prime_rl.orchestrator.records import public_rollout_record, public_task_record


def test_privileged_task_fields_are_redacted_without_mutating_rollout():
    task_record = {
        "idx": 7,
        "prompt": "public prompt",
        "value_function_prompt": "private oracle",
        "nested": {"value_function_prompt": "private nested oracle", "kept": True},
    }
    rollout = SimpleNamespace(to_record=lambda: {"task": task_record.copy(), "reward": 1.0})
    task = SimpleNamespace(model_dump=lambda **_kwargs: task_record.copy())

    assert public_task_record(task) == {
        "idx": 7,
        "prompt": "public prompt",
        "nested": {"kept": True},
    }
    assert public_rollout_record(rollout) == {
        "task": {"idx": 7, "prompt": "public prompt", "nested": {"kept": True}},
        "reward": 1.0,
    }
    assert task_record["value_function_prompt"] == "private oracle"
