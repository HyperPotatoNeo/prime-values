from __future__ import annotations

from typing import Any

_PRIVILEGED_TASK_FIELDS = frozenset({"value_function_prompt"})


def redact_task_record(value: Any) -> Any:
    """Return a JSON-like task value without critic-only task fields."""
    if isinstance(value, dict):
        return {key: redact_task_record(item) for key, item in value.items() if key not in _PRIVILEGED_TASK_FIELDS}
    if isinstance(value, list):
        return [redact_task_record(item) for item in value]
    return value


def public_task_record(task) -> dict[str, Any]:
    return redact_task_record(task.model_dump(mode="json"))


def public_rollout_record(rollout) -> dict[str, Any]:
    record = rollout.to_record()
    if "task" in record:
        record["task"] = redact_task_record(record["task"])
    return record
