"""Compact, full-normalized sampler head payload (capture does not truncate sampling)."""

import base64

import numpy as np


def encode_sampler_head(output, k: int) -> dict:
    if output.logprobs is None or len(output.logprobs) != len(output.token_ids):
        raise ValueError("Missing per-token sampler logprobs")
    ids, logps = [], []
    for row in output.logprobs:
        head = sorted(row.items(), key=lambda item: item[1].rank)[:k]
        if len(head) != k or any(value.rank != rank + 1 for rank, (_, value) in enumerate(head)):
            raise ValueError("Sampler did not return the complete ranked top-k head")
        ids.append([token for token, _ in head])
        logps.append([value.logprob for _, value in head])
    a = np.asarray(ids, dtype="<i4").reshape(-1, k)
    b = np.asarray(logps, dtype="<f4").reshape(-1, k)
    if not np.isfinite(b).all() or (b > 1e-5).any():
        raise ValueError("Invalid sampler head logprobs")
    return {
        "shape": list(a.shape),
        "ids": base64.b64encode(a.tobytes()).decode(),
        "logprobs": base64.b64encode(b.tobytes()).decode(),
    }
