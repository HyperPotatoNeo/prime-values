import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prime_rl.orchestrator.train_sink import TrainSink
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
    sink.value_publisher = None
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
    sink = TrainSink.__new__(TrainSink)
    sink.value_publisher = MagicMock()
    sink.value_publisher.publish.return_value = True
    sink.value_batch_id = 0
    sink.pending_value_rollouts = []
    sink.config = SimpleNamespace(value_function=SimpleNamespace(batch_size=4, model=SimpleNamespace(seq_len=8)))
    rollouts = [
        _rollout(policy_version=2, value_version=4),
        _rollout(policy_version=2, value_version=5),
        _rollout(num_samples=2, policy_version=3, value_version=5),
        _rollout(policy_version=3, value_version=6),
        _rollout(policy_version=3, value_version=6),
        _rollout(policy_version=3, value_version=6),
    ]

    sink.enqueue_value_rollouts(rollouts[:3])
    sink.value_publisher.publish.assert_not_called()
    sink.enqueue_value_rollouts(rollouts[3:])

    sink.value_publisher.publish.assert_called_once()
    batch = sink.value_publisher.publish.call_args.args[0]
    assert batch.num_rollouts == 4
    assert len(batch.samples) == 5
    assert batch.policy_version_min == 2
    assert batch.policy_version_max == 3
    assert batch.value_version_min == 4
    assert batch.value_version_max == 6
    assert len(sink.pending_value_rollouts) == 2
    assert [item.policy_version for item in sink.pending_value_rollouts] == [3, 3]


def test_value_batch_rejects_rollout_without_targets():
    sink = TrainSink.__new__(TrainSink)
    sink.value_publisher = MagicMock()
    sink.value_batch_id = 0
    sink.pending_value_rollouts = []
    sink.config = SimpleNamespace(value_function=SimpleNamespace(batch_size=4, model=SimpleNamespace(seq_len=8)))
    missing = _rollout()
    missing.value_returns = None

    with pytest.raises(RuntimeError, match="lambda-return targets"):
        sink.enqueue_value_rollouts([_rollout(), missing])
    assert sink.pending_value_rollouts == []


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
        sink.pending_value_rollouts = [MagicMock()]
        await sink.stop()

        assert task.cancelled()
        assert sink.scoring_tasks == {}
        assert sink.pending_value_rollouts == []

    asyncio.run(run_test())
