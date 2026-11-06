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


def rollout_monitor_sample(rollout) -> dict[str, Any]:
    """The existing Prime sample columns, derived from the completed owned rollout.

    This small presentation adapter is local because native verifier no longer
    exports its former platform push formatter. Private task fields stay redacted.
    """
    branches = rollout.branches

    def messages(branch):
        return [message.model_dump(mode="json", exclude_none=True) for message in branch.messages]

    timing = rollout.timing.model_dump(mode="json", exclude_none=True)
    timing["generation"] = timing.pop("agent")
    return {
        "example_id": rollout.task.data.idx,
        "task": redact_task_record(rollout.task.data.model_dump(mode="json", exclude_none=True)),
        "completion": messages(branches[-1]) if branches else [],
        "trajectory": [
            {
                "messages": messages(branch),
                "num_input_tokens": branch.num_total_tokens - branch.num_output_tokens,
                "num_output_tokens": branch.num_output_tokens,
            }
            for branch in branches
        ],
        "reward": rollout.reward,
        "metrics": {name: value for name, value in rollout.metrics.items() if value is not None},
        "timing": timing,
    }
