from typing import Literal


def updates_for_batch(
    *,
    value_version: int,
    warmup_updates: int,
    updates_per_batch: int,
) -> int:
    return 1 if value_version < warmup_updates else updates_per_batch


def choose_next_operation(
    *,
    has_inference: bool,
    has_training: bool,
    last_operation: Literal["infer", "train"] | None,
) -> Literal["infer", "train"] | None:
    if has_inference and has_training:
        return "train" if last_operation == "infer" else "infer"
    if has_inference:
        return "infer"
    if has_training:
        return "train"
    return None
