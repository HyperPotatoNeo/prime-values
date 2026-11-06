"""Compact optional comparable-policy-step carrier; never a loss input."""

import numpy as np

from prime_rl.transport.types import EncodedTensor


def decode_policy_versions(payload: EncodedTensor, num_tokens: int) -> np.ndarray:
    if (
        payload.dtype != "int64"
        or payload.shape != [num_tokens]
        or any(type(size) is not int for size in payload.shape)
        or len(payload.data) != num_tokens * 8
    ):
        raise ValueError("token_policy_versions must be int64 with exactly one value per physical token")
    values = np.frombuffer(payload.data, dtype=np.dtype("<i8"))
    if np.any(values < -1):
        raise ValueError("token_policy_versions values must be -1 (unknown) or a nonnegative step")
    return values


def encode_policy_versions(values: np.ndarray) -> EncodedTensor | None:
    if values.dtype != np.dtype("int64") or values.ndim != 1 or np.any(values < -1):
        raise ValueError("policy versions require a one-dimensional int64 array with values >= -1")
    if not np.any(values >= 0):
        return None
    return EncodedTensor(dtype="int64", shape=[len(values)], data=values.astype("<i8", copy=False).tobytes())
