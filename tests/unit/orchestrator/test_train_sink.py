import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prime_rl.orchestrator.train_sink import TrainSink, _ValueBatchAccumulator
from prime_rl.transport import TrainingSample


def _sample() -> TrainingSample:
    return TrainingSample(
        token_ids=[1, 2],
        mask=[False, True],
        logprobs=[0.0, -0.1],
        temperatures=[],
        env_name="test-env",
    )


def _rollout(
    *,
    has_error: bool = False,
    with_sample: bool = True,
    num_samples: int = 1,
    policy_version: int = 0,
    value_version: int = 0,
):
    samples = [_sample() for _ in range(num_samples)] if with_sample and not has_error else []
    return SimpleNamespace(
        env_name="test-env",
        task=SimpleNamespace(idx=0),
        has_error=has_error,
        samples=samples,
        reward=1.0,
        value_returns=[[0.0, 1.0] for _ in samples] if samples else None,
        value_version=value_version if samples else None,
        policy_version=policy_version,
        num_total_tokens=2,
        is_filtered=False,
        filter_results={},
    )


def _group_sink(group, *, minimum_group_size: int) -> tuple[TrainSink, SimpleNamespace]:
    algorithm = SimpleNamespace(minimum_group_size=minimum_group_size, finalize_group=AsyncMock())
    env = SimpleNamespace(
        algorithm=algorithm,
        requires_group_scoring=False,
        sampling_args={"temperature": 1.0},
    )
    sink = TrainSink.__new__(TrainSink)
    sink.pending_groups = {uuid.UUID(int=0): group}
    sink.scoring_tasks = {}
    sink.train_envs = SimpleNamespace(get=lambda _name: env)
    sink._value_batch_accumulator = None
    sink.pre_filters = []
    sink.pending_batch = []
    sink.token_batch_size = None
    sink.pending_tokens = 0
    sink.pre_filter_seen = 0
    sink.pre_filter_dropped = 0
    sink.pre_filter_dropped_by_name = {}
    return sink, env


def test_incomplete_leave_one_out_group_is_dropped():
    async def run_test() -> None:
        survivor = _rollout()
        sink, env = _group_sink([survivor, _rollout(has_error=True)], minimum_group_size=2)

        await sink.process_group(uuid.UUID(int=0))

        env.algorithm.finalize_group.assert_not_awaited()
        assert sink.pending_batch == []

    asyncio.run(run_test())


def test_partial_leave_one_out_group_with_two_survivors_is_scored():
    async def run_test() -> None:
        survivors = [_rollout(), _rollout()]
        sink, env = _group_sink([*survivors, _rollout(has_error=True)], minimum_group_size=2)

        await sink.process_group(uuid.UUID(int=0))

        env.algorithm.finalize_group.assert_awaited_once_with(survivors)
        assert sink.pending_batch == survivors

    asyncio.run(run_test())


def test_value_rollouts_are_accumulated_into_exact_size_batches():
    publisher = MagicMock()
    publisher.publish.return_value = True
    value_batches = _ValueBatchAccumulator(publisher, batch_size=4, seq_len=8)
    rollouts = [
        _rollout(policy_version=2, value_version=4),
        _rollout(policy_version=2, value_version=5),
        _rollout(num_samples=2, policy_version=3, value_version=5),
        _rollout(policy_version=3, value_version=6),
        _rollout(policy_version=3, value_version=6),
        _rollout(policy_version=3, value_version=6),
    ]
    rollouts[4].samples[0].token_ids[0] = 50
    rollouts[5].samples[0].token_ids[0] = 60

    value_batches.add(rollouts[:3])
    publisher.publish.assert_not_called()
    assert value_batches.progress() == (3, 4)
    value_batches.add(rollouts[3:])

    publisher.publish.assert_called_once()
    batch = publisher.publish.call_args.args[0]
    assert batch.batch_id == 0
    assert batch.num_rollouts == 4
    assert len(batch.samples) == 5
    assert batch.policy_version_min == 2
    assert batch.policy_version_max == 3
    assert batch.value_version_min == 4
    assert batch.value_version_max == 6
    assert value_batches.progress() == (2, 4)

    new_rollouts = [
        _rollout(policy_version=8, value_version=8),
        _rollout(policy_version=9, value_version=9),
    ]
    new_rollouts[0].samples[0].token_ids[0] = 80
    new_rollouts[1].samples[0].token_ids[0] = 90
    value_batches.add(new_rollouts)

    assert publisher.publish.call_count == 2
    overflow_batch = publisher.publish.call_args_list[1].args[0]
    assert overflow_batch.batch_id == 1
    assert overflow_batch.policy_version_min == 3
    assert overflow_batch.policy_version_max == 9
    assert overflow_batch.value_version_min == 6
    assert overflow_batch.value_version_max == 9
    assert [sample.token_ids[0] for sample in overflow_batch.samples] == [50, 60, 80, 90]
    assert value_batches.progress() == (0, 4)


def test_value_batch_rejects_rollout_without_targets():
    value_batches = _ValueBatchAccumulator(MagicMock(), batch_size=4, seq_len=8)
    missing = _rollout()
    missing.value_returns = None

    with pytest.raises(RuntimeError, match="lambda-return targets"):
        value_batches.add([_rollout(), missing])
    assert value_batches.progress() == (0, 4)


def test_value_batch_preserves_drop_and_publish_exception_semantics():
    publisher = MagicMock()
    publisher.publish.side_effect = [False, RuntimeError("publish failed"), True]
    value_batches = _ValueBatchAccumulator(publisher, batch_size=1, seq_len=8)

    value_batches.add([_rollout(policy_version=1)])
    with pytest.raises(RuntimeError, match="publish failed"):
        value_batches.add([_rollout(policy_version=2)])
    assert value_batches.progress() == (0, 1)
    value_batches.add([_rollout(policy_version=3)])

    batches = [call.args[0] for call in publisher.publish.call_args_list]
    assert [batch.batch_id for batch in batches] == [0, 1, 1]
    assert [batch.policy_version_min for batch in batches] == [1, 2, 3]


def test_value_batch_truncates_and_copies_sample_data():
    publisher = MagicMock()
    publisher.publish.return_value = True
    value_batches = _ValueBatchAccumulator(publisher, batch_size=1, seq_len=1)
    rollout = _rollout()
    sample = rollout.samples[0]
    targets = rollout.value_returns[0]

    value_batches.add([rollout])

    value_sample = publisher.publish.call_args.args[0].samples[0]
    assert value_sample.token_ids == [1]
    assert value_sample.mask == [False]
    assert value_sample.targets == [0.0]
    assert value_sample.token_ids is not sample.token_ids
    assert value_sample.mask is not sample.mask
    assert value_sample.targets is not targets


def test_value_batch_ignores_rollouts_when_all_targets_are_absent():
    publisher = MagicMock()
    value_batches = _ValueBatchAccumulator(publisher, batch_size=4, seq_len=8)
    rollout = _rollout()
    rollout.value_returns = None

    value_batches.add([_rollout(with_sample=False), rollout])

    publisher.publish.assert_not_called()
    assert value_batches.progress() == (0, 4)


def test_stop_cancels_scoring_for_incomplete_groups():
    async def run_test() -> None:
        started = asyncio.Event()

        async def score() -> None:
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(score())
        await started.wait()

        sink = TrainSink.__new__(TrainSink)
        sink.scoring_tasks = {1: task}
        sink._value_batch_accumulator = _ValueBatchAccumulator(MagicMock(), batch_size=4, seq_len=8)
        sink._value_batch_accumulator.add([_rollout()])
        await sink.stop()

        assert task.cancelled()
        assert sink.scoring_tasks == {}
        assert sink._value_batch_accumulator.progress() == (0, 4)

    asyncio.run(run_test())
