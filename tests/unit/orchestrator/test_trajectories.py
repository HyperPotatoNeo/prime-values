import verifiers.v1 as vf

from prime_rl.orchestrator.trajectories import (
    TrainableSpan,
    materialize_training_data,
    trace_to_samples,
)


def _node(
    token_ids: list[int],
    *,
    parent: int | None,
    sampled: bool = False,
    mask: list[bool] | None = None,
) -> vf.MessageNode:
    token_mask = mask if mask is not None else [sampled] * len(token_ids)
    message = vf.AssistantMessage(content=f"a{token_ids}") if sampled else vf.UserMessage(content=f"u{token_ids}")
    return vf.MessageNode(
        parent=parent,
        message=message,
        sampled=sampled,
        token_ids=token_ids,
        mask=token_mask,
        logprobs=[-0.1] * sum(token_mask),
    )


def _trace(nodes: list[vf.MessageNode]) -> vf.Trace:
    return vf.Trace(task=vf.Task(idx=0, prompt="test"), nodes=nodes)


def test_linear_materialization_preserves_samples_without_layout():
    trace = _trace(
        [
            _node([0], parent=None),
            _node([1, 2], parent=0, sampled=True),
        ]
    )

    materialization = materialize_training_data(trace, env_name="test")

    assert materialization.layout is None
    assert materialization.samples == trace_to_samples(trace, env_name="test")
    assert materialization.samples[0].token_ids == [0, 1, 2]
    assert materialization.samples[0].mask == [False, True, True]


def test_layout_uses_generation_order_when_leaf_order_differs():
    # Generate A, generate sibling B, then extend A. Final leaf/sample order is B, A2,
    # while sampled-node generation order is A, B, A2.
    trace = _trace(
        [
            _node([0], parent=None),
            _node([10], parent=0, sampled=True),
            _node([20], parent=0),
            _node([30], parent=2, sampled=True),
            _node([40], parent=1),
            _node([50, 51], parent=4, sampled=True),
        ]
    )

    materialization = materialize_training_data(trace)

    assert [sample.token_ids for sample in materialization.samples] == [
        [0, 20, 30],
        [0, 10, 40, 50, 51],
    ]
    assert [sample.mask for sample in materialization.samples] == [
        [False, False, True],
        [False, True, False, True, True],
    ]
    assert materialization.layout is not None
    assert materialization.layout.sample_branch_indices == (0, 1)
    assert materialization.layout.sample_node_indices == ((0, 2, 3), (0, 1, 4, 5))
    assert materialization.layout.trainable_spans == (
        TrainableSpan(
            node_index=1,
            sample_index=1,
            node_start=0,
            sample_start=1,
            length=1,
        ),
        TrainableSpan(
            node_index=3,
            sample_index=0,
            node_start=0,
            sample_start=2,
            length=1,
        ),
        TrainableSpan(
            node_index=5,
            sample_index=1,
            node_start=0,
            sample_start=3,
            length=2,
        ),
    )


def test_layout_owns_shared_sampled_prefix_exactly_once():
    trace = _trace(
        [
            _node([0], parent=None),
            _node([10, 11], parent=0, sampled=True),
            _node([20], parent=1),
            _node([30], parent=2, sampled=True),
            _node([40], parent=1),
            _node([50], parent=4, sampled=True),
        ]
    )

    materialization = materialize_training_data(trace)

    assert [sample.mask for sample in materialization.samples] == [
        [False, True, True, False, True],
        [False, False, False, False, True],
    ]
    assert materialization.layout is not None
    assert [span.node_index for span in materialization.layout.trainable_spans] == [
        1,
        3,
        5,
    ]
    assert sum(span.length for span in materialization.layout.trainable_spans) == 4


def test_layout_skips_branch_without_unique_trainable_tokens():
    trace = _trace(
        [
            _node([0], parent=None),
            _node([10], parent=0, sampled=True),
            _node([20], parent=1),
            _node([30], parent=2, sampled=True),
            _node([40], parent=1),
        ]
    )

    materialization = materialize_training_data(trace)

    assert [sample.token_ids for sample in materialization.samples] == [[0, 10, 20, 30]]
    assert materialization.layout is not None
    assert materialization.layout.sample_branch_indices == (0,)
    assert materialization.layout.sample_node_indices == ((0, 1, 2, 3),)
    assert [span.sample_index for span in materialization.layout.trainable_spans] == [
        0,
        0,
    ]


def test_layout_supports_noncontiguous_trainable_masks():
    trace = _trace(
        [
            _node([0], parent=None),
            _node(
                [10, 11, 12, 13],
                parent=0,
                sampled=True,
                mask=[False, True, False, True],
            ),
            _node([20], parent=0),
            _node([30], parent=2, sampled=True),
        ]
    )

    layout = materialize_training_data(trace).layout

    assert layout is not None
    assert layout.trainable_spans[:2] == (
        TrainableSpan(
            node_index=1,
            sample_index=0,
            node_start=1,
            sample_start=2,
            length=1,
        ),
        TrainableSpan(
            node_index=1,
            sample_index=0,
            node_start=3,
            sample_start=4,
            length=1,
        ),
    )
