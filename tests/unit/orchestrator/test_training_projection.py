"""Independent boundary checks for an explicit post-response training view."""

import json
import pickle

import numpy as np
import pytest
import verifiers.v1 as vf
from pydantic import ValidationError
from renderers.base import MultiModalData

from prime_rl.configs.orchestrator import (
    EvalEnvConfig,
    OrchestratorConfig,
    SingleTurnTrainingProjectionConfig,
    TrainEnvConfig,
)
from prime_rl.orchestrator.episodes import UnsupportedEpisodeError, from_episode, project_training_trace
from prime_rl.orchestrator.filters import GibberishFilter, RepetitionFilter
from prime_rl.orchestrator.trajectories import materialize_training_data
from prime_rl.value.trajectory import snapshot_value_trajectory


def _trace():
    # Four prompt tokens: three in the user node and the assistant scaffold.
    # Three actual sampled tokens; a low-probability tail and confident suffix
    # make both token filters sensitive to the training visibility boundary.
    return vf.WireTrace(
        task=vf.TraceTask(type="fixture", data=vf.WireTaskData(idx=9, prompt="question")),
        nodes=[
            vf.MessageNode(
                message=vf.UserMessage(content="question"),
                token_ids=[1, 2, 3],
                mask=[False] * 3,
                is_content=[False, True, True],
            ),
            vf.MessageNode(
                parent=0,
                message=vf.AssistantMessage(content="full answer including clipped tail"),
                token_ids=[4, 10, 1000, 11],
                mask=[False, True, True, True],
                is_content=[False, True, True, True],
                sampled=True,
                logprobs=[-0.1, -8.0, -0.1],
            ),
        ],
        calls=[vf.ModelCall(node=1, finish_reason="stop", usage=vf.Usage(prompt_tokens=4, completion_tokens=3))],
        rewards={"score": 0.75},
        ok=True,
        is_completed=True,
        stop_condition="task_complete",
        info={"environment_owned": {"keep": [1]}},
    )


@pytest.mark.parametrize("limit,expected_length,actions", [(8, 7, 3), (7, 7, 3), (6, 6, 2), (4, 4, 0), (3, 3, 0)])
def test_projection_preserves_physical_response_but_changes_visible_actions(limit, expected_length, actions):
    trace = _trace()
    raw_bytes = trace.model_dump_json()
    projected = project_training_trace(trace, max_seq_len=limit)
    assert trace.model_dump_json() == raw_bytes
    assert projected is not trace
    assert projected.branches[0].token_ids == [1, 2, 3, 4, 10, 1000, 11][:limit]
    assert len(projected.branches[0].token_ids) == expected_length
    assert sum(projected.branches[0].sampled_mask) == actions
    assert projected.branches[0].logprobs == [0.0, 0.0, 0.0, 0.0, -0.1, -8.0, -0.1][:limit]
    assert projected.is_truncated is (limit < 7)
    assert projected.is_completed == trace.is_completed
    assert projected.reward == trace.reward
    assert projected.num_total_tokens == trace.num_total_tokens == 7
    assert projected.num_output_tokens == trace.num_output_tokens == 3
    assert projected.calls[0].model_dump() == trace.calls[0].model_dump()
    assert [node.message.model_dump() for node in projected.nodes] == [
        node.message.model_dump() for node in trace.nodes
    ]
    assert "prime_rl_training_projection" in projected.info
    assert "prime_rl_training_projection" not in trace.info
    materialized = materialize_training_data(projected, capture_sampled_masks=True)
    assert len(materialized.samples) == int(actions > 0)
    for source_node, projected_node in zip(trace.nodes, projected.nodes, strict=True):
        assert projected_node.token_ids is not source_node.token_ids
        assert projected_node.mask is not source_node.mask
        assert projected_node.logprobs is not source_node.logprobs
        assert projected_node.is_content is not source_node.is_content
    projected.nodes[-1].token_ids.clear()
    assert trace.model_dump_json() == raw_bytes


def test_from_episode_owns_projected_arrays_and_keeps_source_and_value_snapshot():
    episode = vf.WireEpisode(id="physical", env="fixture", ok=True, traces=[_trace()])
    raw = episode.model_dump_json()
    no_projection = from_episode(episode)
    assert len(no_projection.branches[0].token_ids) == 7
    assert not no_projection.is_truncated
    rollout = from_episode(episode, training_projection=SingleTurnTrainingProjectionConfig(max_seq_len=5))
    assert episode.model_dump_json() == raw
    assert rollout.source_physical_trace.model_dump_json() == episode.traces[0].model_dump_json()
    record = rollout.model_dump(mode="json")
    assert "source_physical_trace" not in record
    assert "prime_rl_training_projection" in record["info"]
    materialized = materialize_training_data(rollout, capture_sampled_masks=True)
    owned = snapshot_value_trajectory(
        materialized.samples,
        sampled_masks=materialized.sampled_masks,
        layout=materialized.layout,
        reward=rollout.reward,
    )
    rollout.nodes.clear()
    materialized.samples[0].token_ids.clear()
    assert episode.model_dump_json() == raw
    assert owned.samples[0].token_ids == (1, 2, 3, 4, 10)
    assert owned.samples[0].mask == (False, False, False, False, True)
    assert owned.reward == 0.75


def test_token_filters_use_same_projected_boundary_as_materializer():
    physical = _trace()
    gibberish = GibberishFilter("gibberish", token_id_threshold=100, logprob_threshold=-5.0)
    assert gibberish.check(physical).detected
    assert not gibberish.check(project_training_trace(physical, max_seq_len=5)).detected
    assert gibberish.check(project_training_trace(physical, max_seq_len=6)).detected
    physical.nodes[-1].logprobs = [-0.1] * 3
    repetition = RepetitionFilter("repetition", window=2, logprob_threshold=-0.5)
    assert repetition.check(physical).detected
    assert not repetition.check(project_training_trace(physical, max_seq_len=5)).detected
    assert repetition.check(project_training_trace(physical, max_seq_len=6)).detected


@pytest.mark.parametrize(
    "malformation",
    [
        "mask_length",
        "logprob_length",
        "content_length",
        "prompt_action",
        "parent",
        "two_calls",
        "missing_call",
        "wrong_call_node",
        "failed_call",
        "post_response",
        "second_sampled",
        "routed",
        "multimodal",
        "kept_tokens",
        "tools",
    ],
)
def test_projection_rejects_unsupported_or_malformed_source_before_mutation(malformation):
    trace = _trace()
    if malformation == "mask_length":
        trace.nodes[-1].mask.pop()
    elif malformation == "logprob_length":
        trace.nodes[-1].logprobs.pop()
    elif malformation == "content_length":
        trace.nodes[-1].is_content.pop()
    elif malformation == "prompt_action":
        trace.nodes[0].mask[1] = True
        trace.nodes[0].logprobs = [-0.5]
    elif malformation == "parent":
        trace.nodes[-1].parent = None
    elif malformation == "two_calls":
        trace.calls.append(trace.calls[0].model_copy(deep=True))
    elif malformation == "missing_call":
        trace.calls.clear()
    elif malformation == "wrong_call_node":
        trace.calls[0].node = 0
    elif malformation == "failed_call":
        trace.calls[0].error = vf.Error(type="ProviderFailure", message="failed")
    elif malformation == "post_response":
        trace.nodes.append(vf.MessageNode(parent=1, message=vf.UserMessage(content="later observation")))
    elif malformation == "second_sampled":
        trace.nodes[0].sampled = True
    elif malformation == "routed":
        trace.nodes[-1].routed_experts = np.zeros((4, 1, 1), dtype=np.uint8)
    elif malformation == "multimodal":
        trace.nodes[0].multi_modal_data = MultiModalData(mm_hashes={"image": ["fixture"]})
    elif malformation == "kept_tokens":
        trace.nodes[-1].kept_tokens = vf.KeptTokens(
            ids=np.array([10, 1000, 11], dtype=np.int32), counts=np.ones(3, dtype=np.int32)
        )
    elif malformation == "tools":
        trace.nodes[-1].message.tool_calls = [vf.ToolCall(id="call", name="tool", arguments="{}")]
    before = pickle.dumps(trace.model_dump(mode="python"))
    with pytest.raises(UnsupportedEpisodeError):
        project_training_trace(trace, max_seq_len=5)
    assert pickle.dumps(trace.model_dump(mode="python")) == before


def test_projection_keeps_sampled_logprobs_aligned_when_mask_has_holes():
    trace = _trace()
    trace.nodes[-1].mask = [False, True, False, True]
    trace.nodes[-1].logprobs = [-0.2, -0.7]
    projected = project_training_trace(trace, max_seq_len=6)
    assert projected.branches[0].sampled_mask == [False, False, False, False, True, False]
    assert projected.branches[0].logprobs == [0.0, 0.0, 0.0, 0.0, -0.2, 0.0]


def test_failed_empty_or_frozen_episodes_keep_existing_failure_and_eligibility_semantics():
    config = SingleTurnTrainingProjectionConfig(max_seq_len=5)
    trace = _trace()
    trace.calls.clear()  # Would be inadmissible if projection incorrectly ran.
    episode = vf.WireEpisode(ok=False, traces=[trace])
    rollout = from_episode(episode, training_projection=config)
    assert rollout.has_error and rollout.error.type == "EpisodeFailed"
    assert "prime_rl_training_projection" not in rollout.info
    trace.ok = False
    trace.errors = [vf.Error(type="OriginalFailure", message="provider failed")]
    rollout = from_episode(vf.WireEpisode(ok=True, traces=[trace]), training_projection=config)
    assert rollout.has_error and rollout.error.type == "OriginalFailure"
    empty = from_episode(vf.WireEpisode(ok=False), task_idx=19, training_projection=config)
    assert empty.has_error and empty.task.data.idx == 19
    trace.ok = True
    trace.errors = []
    trace.agent = vf.AgentInfo(model="frozen", trainable=False)
    frozen = from_episode(vf.WireEpisode(ok=True, traces=[trace]), training_projection=config)
    assert not frozen.trainable and not frozen.has_error
    assert "prime_rl_training_projection" not in frozen.info


def test_projection_config_is_explicit_native_training_only_and_roundtrips():
    native = {"env": {"taskset": {"id": "rg-mix-v1", "dataset_path": "/unused"}}}
    assert TrainEnvConfig(**native).training_projection is None
    with pytest.raises(ValidationError):
        TrainEnvConfig(id="reverse-text", training_projection={"type": "single_turn"})
    with pytest.raises(ValidationError):
        EvalEnvConfig(**native, training_projection={"type": "single_turn"})
    for limit in (0, -1, True, 1.5, "5"):
        with pytest.raises(ValidationError):
            SingleTurnTrainingProjectionConfig(max_seq_len=limit)
    config = OrchestratorConfig(
        model={"name": "fixture"},
        renderer={"name": "qwen3"},
        seq_len=64,
        train={"env": [native | {"training_projection": {"type": "single_turn"}}]},
    )
    assert config.train.env[0].training_projection.max_seq_len == 64
    restored = OrchestratorConfig.model_validate(json.loads(config.model_dump_json()))
    assert restored.train.env[0].training_projection == config.train.env[0].training_projection
    explicit = OrchestratorConfig(
        model={"name": "fixture"},
        renderer={"name": "qwen3"},
        seq_len=64,
        train={"env": [native | {"training_projection": {"type": "single_turn", "max_seq_len": 32}}]},
    )
    assert explicit.train.env[0].training_projection.max_seq_len == 32
