from typing import Literal


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
