"""Independent physical-coordinate and observation-only provenance contracts."""

import msgspec
import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1.configs.agent import WireAgentConfig
from verifiers.v1.serve.delta import dump
from verifiers.v1.types import WeightVersionSpan

from prime_rl.configs.orchestrator import SingleTurnTrainingProjectionConfig
from prime_rl.orchestrator.episodes import UnsupportedEpisodeError, from_episode
from prime_rl.orchestrator.provenance import node_policy_versions, validate_weight_versions
from prime_rl.orchestrator.trajectories import materialize_training_data
from prime_rl.transport.provenance import decode_policy_versions
from prime_rl.value.trajectory import snapshot_value_trajectory


def _node(parent, ids, mask, spans=None):
    sampled = any(mask)
    return vf.MessageNode(
        parent=parent,
        message=vf.AssistantMessage(content="answer") if sampled else vf.UserMessage(content="question"),
        sampled=sampled,
        token_ids=ids,
        mask=mask,
        logprobs=[-0.125] * sum(mask),
        weight_versions=None if spans is None else [WeightVersionSpan(version=v, start=a, end=b) for v, a, b in spans],
    )


def _episode(nodes):
    trace = vf.WireTrace(
        task=vf.TraceTask(type="fixture", data=vf.WireTaskData(idx=1, prompt="question")),
        agent=vf.AgentInfo(config=WireAgentConfig(model="fixture", runtime={"type": "subprocess"})),
        nodes=nodes,
        calls=[vf.ModelCall(node=i, finish_reason="stop") for i, n in enumerate(nodes) if n.sampled],
        ok=True,
        is_completed=True,
    )
    return vf.WireEpisode(task=trace.task, env=vf.EnvInfo(id="fixture"), traces=[trace], ok=True)


def test_completion_ordinals_follow_original_mask_not_suffix_or_token_identity():
    node = _node(None, [8] * 8, [False, True, False, True, True, False, True, True], [("3", 0, 2), ("4", 3, 5)])
    before = dump(node)
    assert node_policy_versions(node).tolist() == [-1, 3, -1, 3, -1, -1, 4, 4]
    assert dump(node) == before
    # A zero-sized stopped chunk is valid but attributes no token.
    node.weight_versions.append(WeightVersionSpan(version="9", start=5, end=5))
    assert node_policy_versions(node).tolist() == [-1, 3, -1, 3, -1, -1, 4, 4]


def test_raw_opaque_labels_survive_but_only_canonical_step_ids_are_comparable():
    labels = ["0", str(2**63 - 1), "default", "00", "+3", "3 ", "-1", "３", str(2**63)]
    node = _node(None, list(range(len(labels))), [True] * len(labels), [(v, i, i + 1) for i, v in enumerate(labels)])
    before = dump(node)
    assert node_policy_versions(node).tolist() == [0, 2**63 - 1, -1, -1, -1, -1, -1, -1, -1]
    assert dump(node) == before
    assert [x.version for x in node.weight_versions] == labels
    for missing in [None, []]:
        node.weight_versions = missing
        assert node_policy_versions(node).tolist() == [-1] * len(labels)


@pytest.mark.parametrize(
    "intervals",
    [
        [(-1, 1)],
        [(0, 4)],
        [(2, 1)],
        [(0, 2), (1, 3)],
        [(2, 3), (0, 1)],
        [(False, 1)],
        [(0, 1.5)],
    ],
)
def test_malformed_geometry_fails_without_downgrading_to_unknown_or_mutating(intervals):
    node = _node(None, [1, 2, 3], [True, True, True])
    node.weight_versions = [WeightVersionSpan.model_construct(version="opaque", start=a, end=b) for a, b in intervals]
    before = dump(node)
    with pytest.raises(ValueError):
        validate_weight_versions(node)
    assert dump(node) == before
    episode = _episode([node])
    with pytest.raises(UnsupportedEpisodeError):
        from_episode(episode)
    assert dump(node) == before


def test_projection_clips_original_sample_ordinals_without_touching_source_trace():
    episode = _episode(
        [
            _node(None, [1, 2], [False, False]),
            _node(0, [3, 4, 5, 6, 7], [False, True, False, True, True], [("3", 0, 2), ("9", 2, 3)]),
        ]
    )
    before = dump(episode)
    rollout = from_episode(episode, training_projection=SingleTurnTrainingProjectionConfig(max_seq_len=5))
    assert rollout.nodes[1].token_ids == [3, 4, 5]
    assert node_policy_versions(rollout.nodes[1]).tolist() == [-1, 3, -1]
    assert dump(episode) == before
    assert dump(rollout.source_physical_trace) == before["traces"][0]
    assert [(s.version, s.start, s.end) for s in rollout.nodes[1].weight_versions] == [("3", 0, 1)]
    empty = from_episode(episode, training_projection=SingleTurnTrainingProjectionConfig(max_seq_len=2))
    assert node_policy_versions(empty.nodes[1]).tolist() == []
    assert dump(episode) == before


def test_shared_prefix_preserves_sampling_labels_independently_of_training_deduplication():
    episode = _episode(
        [
            _node(None, [1], [False]),
            _node(0, [2, 3], [True, True], [("7", 0, 1), ("8", 1, 2)]),
            _node(1, [4], [False]),
            _node(2, [5, 6], [True, True], [("9", 0, 2)]),
            _node(1, [7], [False]),
            _node(4, [8], [True], [("10", 0, 1)]),
        ]
    )
    rollout = from_episode(episode)
    before = dump(rollout)
    live = materialize_training_data(rollout, capture_policy_versions=True, capture_sampled_masks=True)
    frozen = materialize_training_data(rollout, capture_policy_versions=False, capture_sampled_masks=True)
    assert [s.mask for s in live.samples] == [
        [False, True, True, False, True, True],
        [False, False, False, False, True],
    ]
    expected = [[-1, 7, 8, -1, 9, 9], [-1, 7, 8, -1, 10]]
    for actual, absent, labels in zip(live.samples, frozen.samples, expected, strict=True):
        np.testing.assert_array_equal(
            decode_policy_versions(actual.token_policy_versions, len(actual.token_ids)), labels
        )
        assert absent.token_policy_versions is None
        d = msgspec.structs.asdict(actual)
        d["token_policy_versions"] = None
        assert d == msgspec.structs.asdict(absent)
    assert snapshot_value_trajectory(
        live.samples, sampled_masks=live.sampled_masks, layout=live.layout, reward=1
    ) == snapshot_value_trajectory(frozen.samples, sampled_masks=frozen.sampled_masks, layout=frozen.layout, reward=1)
    assert dump(rollout) == before
