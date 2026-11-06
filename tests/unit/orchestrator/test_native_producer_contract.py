"""Independent policy/value contracts across the canonical producer upgrade."""

import pytest
import verifiers.v1 as vf
from rg_mix_v1.taskset import RGMixData, RGMixTask
from verifiers.v1.configs.agent import WireAgentConfig
from verifiers.v1.serve.delta import dump, pack, unpack

from prime_rl.configs.orchestrator import SingleTurnTrainingProjectionConfig
from prime_rl.orchestrator.dispatcher import _failure_rollout
from prime_rl.orchestrator.episodes import UnsupportedEpisodeError, from_episode
from prime_rl.orchestrator.metrics import CustomMetrics
from prime_rl.orchestrator.records import public_rollout_record
from prime_rl.orchestrator.trajectories import materialize_training_data
from prime_rl.orchestrator.types import GroupState
from prime_rl.value.trajectory import snapshot_value_trajectory


def _branching_trace():
    def node(parent, ids, *, sampled=False):
        message = vf.AssistantMessage(content="answer") if sampled else vf.UserMessage(content="question")
        return vf.MessageNode(
            parent=parent,
            message=message,
            sampled=sampled,
            token_ids=ids,
            mask=[sampled] * len(ids),
            logprobs=[-0.123456789121] * len(ids) if sampled else [],
        )

    return vf.WireTrace(
        task=vf.TraceTask(type="fixture", data=vf.WireTaskData(idx=17, prompt="question")),
        agent=vf.AgentInfo(config=WireAgentConfig(model="fixture", runtime={"type": "subprocess"})),
        nodes=[
            node(None, [1, 2, 3, 4]),
            node(0, [5, 6], sampled=True),
            node(1, [7, 8]),
            node(2, [9, 10, 11], sampled=True),
            node(1, [12, 13, 14]),
            node(4, [15, 16, 17, 18], sampled=True),
        ],
        calls=[
            vf.ModelCall(node=1, usage=vf.Usage(prompt_tokens=4, completion_tokens=2), finish_reason="stop"),
            vf.ModelCall(node=3, usage=vf.Usage(prompt_tokens=8, completion_tokens=3), finish_reason="stop"),
            vf.ModelCall(node=5, usage=vf.Usage(prompt_tokens=9, completion_tokens=4), finish_reason="stop"),
        ],
        rewards={"answer": vf.Reward(score=0.75, weight=1.0)},
        ok=True,
        is_completed=True,
    )


def _episode(trace, *, ok=True, errors=()):
    return vf.WireEpisode(task=trace.task, env=vf.EnvInfo(id="fixture"), traces=[trace], ok=ok, errors=list(errors))


def test_plain_shared_prefix_keeps_old_accounting_and_unique_credit_geometry():
    trace = _branching_trace()
    # Independent literal counter oracle: physical last sequences11+13;
    # shared completion2 appears in both branch output counters, as before.
    assert trace.num_total_tokens == 18 and trace.num_input_tokens == 9
    rollout = from_episode(_episode(trace))
    assert (rollout.num_total_tokens, rollout.num_output_tokens, rollout.num_input_tokens) == (24, 11, 13)
    assert rollout.usage == trace.usage
    materialized = materialize_training_data(rollout, capture_sampled_masks=True)
    assert [s.token_ids for s in materialized.samples] == [
        list(range(1, 12)),
        [1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18],
    ]
    assert [sum(s.mask) for s in materialized.samples] == [5, 4]
    assert [sum(mask) for mask in materialized.sampled_masks] == [5, 6]
    assert [span.node_index for span in materialized.layout.trainable_spans] == [1, 3, 5]
    owned = snapshot_value_trajectory(
        materialized.samples,
        sampled_masks=materialized.sampled_masks,
        layout=materialized.layout,
        reward=rollout.reward,
    )
    materialized.samples[0].token_ids.clear()
    assert owned.samples[0].token_ids == tuple(range(1, 12))
    assert sum(sum(s.mask) for s in owned.samples) == 9


def test_msgpack_keeps_physical_coordinates_and_full_precision_token_evidence():
    trace = _branching_trace()
    trace.nodes[1].renderer_token_ids = [99001]
    assert trace.nodes[1].logical_ids != trace.nodes[1].token_ids
    before = from_episode(_episode(trace))
    before_samples = materialize_training_data(before).samples
    received = vf.WireEpisode.model_validate(unpack(pack(dump(_episode(trace)))))
    after = from_episode(received)
    after_samples = materialize_training_data(after).samples
    assert [s.token_ids for s in after_samples] == [s.token_ids for s in before_samples]
    assert [s.logprobs for s in after_samples] == [s.logprobs for s in before_samples]
    assert after.nodes[1].logprobs == [-0.123456789121] * 2
    assert received.traces[0].nodes[1].renderer_token_ids is None
    # The public record is intentionally rounded; it cannot be the exact oracle.
    assert trace.model_dump(mode="json")["nodes"][1]["logprobs"] != trace.nodes[1].logprobs
    # A live renderer coordinate cannot be clipped with a physical-token index.
    # The unprojected path above remains supported; the explicit projection fails.
    trace.nodes = trace.nodes[:2]
    trace.calls = trace.calls[:1]
    with pytest.raises(UnsupportedEpisodeError, match="renderer"):
        from_episode(_episode(trace), training_projection=SingleTurnTrainingProjectionConfig(max_seq_len=5))


def test_weighted_rewards_are_counted_once_and_unscored_metrics_are_not_zeros():
    first = _branching_trace()
    first.rewards = {
        "answer": vf.Reward(score=0.8, weight=0.25),
        "format": vf.Reward(score=1, weight=0),
        "unscored": None,
    }
    first.metrics = {"raw_answer": 0.8, "raw_format": 1, "unscored": None}
    second = _branching_trace()
    second.rewards = {"answer": vf.Reward(score=0, weight=0.25)}
    second.metrics = {"raw_answer": 0, "raw_format": None}
    rollouts = [from_episode(_episode(t)) for t in (first, second)]
    assert [r.reward for r in rollouts] == [0.2, 0.0]
    rewards = CustomMetrics(rollouts, "rewards").to_dict("reward")
    metrics = CustomMetrics(rollouts, "metrics").to_dict("metric")
    assert rewards["reward/answer/mean"] == 0.1
    assert rewards["reward/format/mean"] == 0.0
    assert metrics["metric/raw_answer/mean"] == 0.4
    assert metrics["metric/raw_format/mean"] == 1.0
    assert not any("unscored" in key for key in rewards | metrics)


@pytest.mark.parametrize("kind", ["semantic", "compaction", "compaction_failed", "namespace"])
def test_new_semantic_credit_shapes_fail_without_banning_plain_branches(kind):
    trace = _branching_trace()
    if kind == "semantic":
        trace.nodes[-1].semantic_parents = [vf.ParentLink(node=3, type="subagent_return")]
    elif kind == "compaction":
        trace.nodes[-1].semantic_parents = [vf.ParentLink(node=3, type="compaction_attempt")]
        assert not trace.branches[-1].trainable
    elif kind == "compaction_failed":
        trace.stop_condition = "compaction_failed"
    else:
        trace.nodes[-1].message.tool_calls = [
            vf.ToolCall(id="tool", name="same_name", namespace="private", arguments="{}")
        ]
    before = dump(trace)
    with pytest.raises(UnsupportedEpisodeError):
        from_episode(_episode(trace))
    assert dump(trace) == before


def test_task_owned_private_prefix_survives_typed_wire_but_is_redacted_from_records():
    data = RGMixData(
        idx=17,
        prompt="public question",
        answer="10017",
        variant_id="fixture",
        scorer_entry={"answer": "private scorer answer"},
        value_function_prompt="private critic prefix",
    )
    rebuilt = RGMixTask.data_type().model_validate(data.model_dump(mode="json"))
    assert rebuilt.value_function_prompt == "private critic prefix"
    trace = _branching_trace()
    trace.task = vf.TraceTask(type="RGMixTask", data=rebuilt)
    typed_trace = vf.Trace[RGMixData].model_validate(dump(trace))
    typed_episode = vf.Episode[RGMixData](
        task=typed_trace.task, env=vf.EnvInfo(id="fixture"), traces=[typed_trace], ok=True
    )
    wire = vf.WireEpisode.model_validate(unpack(pack(dump(typed_episode))))
    rollout = from_episode(wire)
    assert rollout.task.data.value_function_prompt == "private critic prefix"
    assert "value_function_prompt" not in public_rollout_record(rollout)["task"]["data"]
    assert rollout.task.data.prompt == "public question"
    assert all("private critic prefix" not in str(n.message.content) for n in rollout.nodes)
    failed = vf.WireEpisode(
        task=wire.task, env=wire.env, ok=False, errors=[vf.Error(type="SetupFailure", message="no trace")]
    )
    marker = from_episode(failed, task_idx=99)
    assert marker.has_error and marker.error.type == "SetupFailure"
    assert marker.task.data.idx == 17 and marker.task.data.value_function_prompt == "private critic prefix"
    assert "value_function_prompt" not in public_rollout_record(marker)["task"]["data"]


def test_recovered_error_history_and_original_trace_count_keep_existing_semantics():
    trace = _branching_trace()
    historical = vf.Error(type="Retryable", message="earlier attempt")
    trace.errors = [historical]
    recovered = from_episode(_episode(trace, errors=[historical]))
    assert not recovered.has_error and recovered.error.type == "Retryable"
    failed = from_episode(_episode(trace, ok=False))
    assert failed.has_error and failed.error.type == "EpisodeFailed"
    frozen = trace.model_copy(deep=True)
    frozen.agent.trainable = False
    episode = _episode(trace)
    episode.traces.append(frozen)
    with pytest.raises(UnsupportedEpisodeError, match="2 original traces"):
        from_episode(episode)


def test_old_framework_limit_markers_retain_training_truncation_meaning():
    for marker in ("context_length", "harness_timeout"):
        trace = _branching_trace()
        trace.stop_condition = marker
        assert not trace.is_truncated  # Target canonical semantics changed.
        rollout = from_episode(_episode(trace))
        assert rollout.is_truncated
        assert rollout.stop_condition == trace.stop_condition == marker


def test_dispatch_failure_markers_retain_owned_private_task_without_aliases():
    payload = {"idx": 17, "prompt": "question", "value_function_prompt": "private prefix", "nested": {"ids": [1, 2]}}
    group = GroupState(
        kind="train",
        env_name="fixture",
        task_idx=17,
        rollouts_to_schedule=1,
        target_rollouts=2,
        task_data=payload,
    )
    cancelled = _failure_rollout(group, 17, cancelled=True)
    assert cancelled.has_error and cancelled.error.type == "Cancelled"
    assert not cancelled.trainable and cancelled.stop_condition == "error"
    assert cancelled.task.data.value_function_prompt == "private prefix"
    assert "value_function_prompt" not in public_rollout_record(cancelled)["task"]["data"]
    payload["nested"]["ids"].append(3)
    assert cancelled.task.data.nested == {"ids": [1, 2]}
    cancelled.task.data.nested["ids"].append(4)
    assert payload["nested"]["ids"] == [1, 2, 3]
    failed = _failure_rollout(None, 23)
    failed.record_error(ValueError("provider failure"))
    failed.stop("error", override=True)
    assert failed.has_error and not failed.trainable
    assert failed.task.data.idx == 23 and failed.error is not None
    assert not materialize_training_data(failed).samples
