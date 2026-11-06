"""Project a complete upstream episode onto the supported single-trace credit unit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf
from verifiers.v1.types import TextContentPart

from prime_rl.orchestrator.types import Rollout

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import SingleTurnTrainingProjectionConfig


class UnsupportedEpisodeError(ValueError):
    """The episode requires credit semantics this runtime does not define."""


def project_training_trace(trace: vf.WireTrace, *, max_seq_len: int) -> vf.WireTrace:
    """Own a single-turn text trace with a capped training-token prefix.

    Messages, rewards and provider calls remain complete. Retaining empty suffix
    nodes preserves the conversation and call indices for peer context and audit.
    Clipping sets the canonical max_total_tokens marker for training truncation;
    the original stop condition and flags remain in the projection receipt.
    """

    def reject(reason: str) -> None:
        raise UnsupportedEpisodeError(f"trace {trace.id!r}: single_turn training projection {reason}")

    if type(max_seq_len) is not int or max_seq_len <= 0:
        reject("requires a resolved positive integer max_seq_len")
    if "prime_rl_training_projection" in trace.info:
        reject("cannot be applied to an already projected trace")
    if not trace.nodes or len(trace.calls) != 1:
        reject("requires one model call and one nonempty linear branch")
    call = trace.calls[0]
    if call.error is not None or call.node != len(trace.nodes) - 1:
        reject("requires one successful call linked to the final node")
    if trace.tools:
        reject("does not support tools")
    sampled_nodes = [index for index, node in enumerate(trace.nodes) if node.sampled]
    if sampled_nodes != [len(trace.nodes) - 1]:
        reject("requires exactly one final sampled assistant node")
    for index, node in enumerate(trace.nodes):
        if node.parent != (index - 1 if index else None):
            reject("requires a single linear branch in node order")
        if not isinstance(node.message, (vf.SystemMessage, vf.UserMessage, vf.AssistantMessage)):
            reject("supports text messages only")
        content = node.message.content
        if content is not None and not isinstance(content, str):
            if not isinstance(content, list) or any(not isinstance(part, TextContentPart) for part in content):
                reject("does not support non-text message content")
        if isinstance(node.message, vf.AssistantMessage) and node.message.tool_calls:
            reject("does not support tool calls")
        if node.sampled and not isinstance(node.message, vf.AssistantMessage):
            reject("requires a sampled assistant response")
        if any(sidecar is not None for sidecar in (node.multi_modal_data, node.routed_experts, node.kept_tokens)):
            reject("does not support multimodal, routing or kept-token sidecars")
        if len(node.mask) != len(node.token_ids) or any(type(mask) is not bool for mask in node.mask):
            reject("requires token-aligned boolean masks")
        if len(node.logprobs) != sum(node.mask):
            reject("requires one logprob per sampled token")
        if node.is_content and len(node.is_content) != len(node.token_ids):
            reject("requires token-aligned content masks when present")
        if not node.sampled and any(node.mask):
            reject("does not support sampled tokens on prompt nodes")

    physical_tokens = sum(len(node.token_ids) for node in trace.nodes)
    physical_sampled_tokens = sum(sum(node.mask) for node in trace.nodes)
    projected = trace.model_copy(deep=True)
    remaining = max_seq_len
    for node in projected.nodes:
        retained = min(len(node.token_ids), remaining)
        node.token_ids = node.token_ids[:retained]
        node.mask = node.mask[:retained]
        node.is_content = node.is_content[:retained]
        node.logprobs = node.logprobs[: sum(node.mask)]
        remaining -= retained
    clipped = physical_tokens > max_seq_len
    if clipped:
        projected.stop_condition = "max_total_tokens"
    projected.info["prime_rl_training_projection"] = {
        "type": "single_turn",
        "record_kind": "projected_training_trace",
        "max_seq_len": max_seq_len,
        "clipped": clipped,
        "source_trace_id": trace.id,
        "original_stop_condition": trace.stop_condition,
        "original_is_completed": trace.is_completed,
        "original_is_truncated": trace.is_truncated,
        "physical_tokens": physical_tokens,
        "physical_sampled_tokens": physical_sampled_tokens,
        "retained_tokens": min(physical_tokens, max_seq_len),
        "retained_sampled_tokens": sum(sum(node.mask) for node in projected.nodes),
    }
    return projected


def from_episode(
    episode: vf.WireEpisode,
    *,
    task_idx: int = -1,
    training_projection: SingleTurnTrainingProjectionConfig | None = None,
) -> Rollout:
    # Selection precedes eligibility: a frozen or failed sibling still makes
    # this a multi-trace episode, with undefined cross-agent value credit.
    if len(episode.traces) > 1:
        raise UnsupportedEpisodeError(
            f"episode {episode.id!r} contains {len(episode.traces)} original traces; "
            "this runtime requires one trace per episode"
        )
    if episode.traces:
        trace = episode.traces[0]
        source_physical_trace = None
        if training_projection is not None and episode.ok and trace.ok and trace.trainable:
            source_physical_trace = trace
            trace = project_training_trace(trace, max_seq_len=training_projection.max_seq_len)
        rollout = Rollout[vf.WireTaskData].model_construct(**dict(trace))
        rollout.source_physical_trace = source_physical_trace
        rollout.errors = list(trace.errors)
        rollout.info = dict(trace.info)
        if not episode.ok:
            rollout.ok = False
            rollout.errors.append(
                episode.error or vf.Error(type="EpisodeFailed", message="The environment episode did not complete")
            )
        if not rollout.ok and not rollout.errors:
            rollout.errors.append(vf.Error(type="TraceFailed", message="The trace did not complete"))
    else:
        rollout = Rollout[vf.WireTaskData](
            task=vf.TraceTask(type="DispatchTask", data=vf.WireTaskData(idx=task_idx)),
            ok=False,
            errors=list(episode.errors) or [vf.Error(type="EmptyEpisode", message="The environment returned no trace")],
        )
    rollout.info["prime_rl_episode"] = {
        "id": episode.id,
        "env": episode.env,
        "ok": episode.ok,
        "errors": [error.model_dump(mode="json", exclude_none=True) for error in episode.errors],
    }
    return rollout
