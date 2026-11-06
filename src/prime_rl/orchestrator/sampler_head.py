"""Align recorded completion heads with branch tokens without retokenizing."""

import base64

import numpy as np

from prime_rl.transport.types import EncodedTensor


def branch_sampler_head(branch):
    present = [n.sampler_head for n in branch.nodes if n.sampler_head is not None]
    if not present:
        return None, None
    k = present[0]["shape"][1]
    ids, logps = [], []
    for node in branch.nodes:
        a = np.zeros((len(node.token_ids), k), dtype="<i4")
        b = np.zeros((len(node.token_ids), k), dtype="<f4")
        mask = np.asarray(node.mask, dtype=bool)
        if mask.any():
            payload = node.sampler_head
            if payload is None or payload["shape"] != [int(mask.sum()), k]:
                raise ValueError("Sampler head missing or misaligned on a sampled node")
            aa = np.frombuffer(base64.b64decode(payload["ids"], validate=True), dtype="<i4").reshape(-1, k)
            bb = np.frombuffer(base64.b64decode(payload["logprobs"], validate=True), dtype="<f4").reshape(-1, k)
            if aa.shape != (int(mask.sum()), k) or bb.shape != aa.shape:
                raise ValueError("Invalid sampler head byte lengths")
            if (aa < 0).any() or not np.isfinite(bb).all() or (bb > 1e-5).any():
                raise ValueError("Invalid sampler head values")
            if (np.diff(np.sort(aa, axis=1), axis=1) == 0).any():
                raise ValueError("Duplicate sampler candidate IDs")
            a[mask], b[mask] = aa, bb
        ids.append(a)
        logps.append(b)

    def encode(rows):
        array = np.concatenate(rows)
        return EncodedTensor(str(array.dtype), list(array.shape), array.tobytes())

    return encode(ids), encode(logps)
