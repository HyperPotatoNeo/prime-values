"""Observed engine labels, independent of credit masks and dispatch versions."""

from __future__ import annotations

import re

import numpy as np
import verifiers.v1 as vf

_STEP_LABEL = re.compile(r"(?:0|[1-9][0-9]*)\Z", re.ASCII)
_MAX_STEP = str(np.iinfo(np.int64).max)


def validate_weight_versions(node: vf.MessageNode) -> None:
    if node.weight_versions is None:
        return
    if len(node.mask) != len(node.token_ids) or any(type(flag) is not bool for flag in node.mask):
        raise ValueError("weight versions require a token-aligned boolean sampling mask")
    if not node.sampled and (node.weight_versions or any(node.mask)):
        raise ValueError("weight versions require a sampled node")
    sampled_count = sum(node.mask)
    previous_end = 0
    for span in node.weight_versions:
        if (
            type(span.start) is not int
            or type(span.end) is not int
            or not isinstance(span.version, str)
            or not 0 <= previous_end <= span.start <= span.end <= sampled_count
        ):
            raise ValueError("weight versions require ordered, nonoverlapping sampled-token intervals")
        previous_end = span.end


def _policy_step(label: str) -> int:
    # The publisher's protocol is str(nonnegative_step), not arbitrary int parsing.
    # Opaque/default/out-of-range labels remain exact on Trace, but are unknown here.
    if not _STEP_LABEL.fullmatch(label):
        return -1
    if len(label) > len(_MAX_STEP) or (len(label) == len(_MAX_STEP) and label > _MAX_STEP):
        return -1
    return int(label)


def node_policy_versions(node: vf.MessageNode) -> np.ndarray:
    """Comparable step IDs at physical positions; context/unknown is -1.

    Shared-prefix copies retain their sampling provenance even when the canonical
    materializer deduplicates their training mask. Never infer identity from text.
    """
    validate_weight_versions(node)
    result = np.full(len(node.token_ids), -1, dtype=np.int64)
    if not node.weight_versions:
        return result
    sampled_positions = np.flatnonzero(node.mask)
    for span in node.weight_versions:
        result[sampled_positions[span.start : span.end]] = _policy_step(span.version)
    return result


def clip_weight_versions(node: vf.MessageNode, retained_sampled: int):
    """Copy raw spans onto a sampled prefix; the input node stays untouched."""
    validate_weight_versions(node)
    if type(retained_sampled) is not int or not 0 <= retained_sampled <= sum(node.mask):
        raise ValueError("invalid retained sampled-token count")
    if node.weight_versions is None:
        return None
    return [
        span.model_copy(update={"end": min(span.end, retained_sampled)})
        for span in node.weight_versions
        if span.start < retained_sampled
    ]
