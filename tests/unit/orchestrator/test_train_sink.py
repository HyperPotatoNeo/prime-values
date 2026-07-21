import asyncio
import uuid
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prime_rl.orchestrator.metrics import TrainRollouts
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.value_context import TokenPrefix
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
        value_prefix=None,
        policy_version=policy_version,
        num_total_tokens=2,
        is_filtered=False,
        filter_results={},
    )


def _group_sink(
    group,
    *,
    minimum_group_size: int,
    value_publisher=None,
) -> tuple[TrainSink, SimpleNamespace]:
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
    sink._value_publisher = value_publisher
    sink._value_seq_len = 8 if value_publisher is not None else None
    sink._next_value_rollout_id = 0
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


def _value_sink(publisher, *, seq_len: int = 8) -> TrainSink:
    sink = TrainSink.__new__(TrainSink)
    sink._value_publisher = publisher
    sink._value_seq_len = seq_len
    sink._next_value_rollout_id = 0
    return sink


def test_value_rollouts_are_published_individually_with_monotonic_ids():
    publisher = MagicMock()
    sink = _value_sink(publisher)
    rollouts = [
        _rollout(policy_version=2, value_version=4),
        _rollout(num_samples=2, policy_version=3, value_version=5),
    ]

    sink._publish_value_rollouts(rollouts)

    published = [call.args[0] for call in publisher.publish.call_args_list]
    assert [rollout.rollout_id for rollout in published] == [0, 1]
    assert [rollout.policy_version for rollout in published] == [2, 3]
    assert [rollout.value_version for rollout in published] == [4, 5]
    assert [len(rollout.samples) for rollout in published] == [1, 2]

    sink._publish_value_rollouts([_rollout(policy_version=8, value_version=9)])
    assert publisher.publish.call_args.args[0].rollout_id == 2


def test_value_rollouts_reject_mixed_target_state_before_publishing():
    publisher = MagicMock()
    sink = _value_sink(publisher)
    missing = _rollout()
    missing.value_returns = None

    with pytest.raises(RuntimeError, match="lambda-return targets"):
        sink._publish_value_rollouts([_rollout(), missing])
    publisher.publish.assert_not_called()


def test_value_rollout_truncates_and_copies_sample_data():
    publisher = MagicMock()
    sink = _value_sink(publisher, seq_len=1)
    rollout = _rollout()
    sample = rollout.samples[0]
    targets = rollout.value_returns[0]

    sink._publish_value_rollouts([rollout])

    value_sample = publisher.publish.call_args.args[0].samples[0]
    assert value_sample.token_ids == [1]
    assert value_sample.mask == [False]
    assert value_sample.targets == [0.0]
    assert value_sample.token_ids is not sample.token_ids
    assert value_sample.mask is not sample.mask
    assert value_sample.targets is not targets


def test_value_rollout_lifts_policy_streams_around_privileged_prefix():
    publisher = MagicMock()
    sink = _value_sink(publisher, seq_len=4)
    rollout = _rollout()
    rollout.value_prefix = TokenPrefix(token_ids=(90, 91), insert_at=1)
    policy_token_ids = list(rollout.samples[0].token_ids)
    policy_mask = list(rollout.samples[0].mask)
    policy_targets = list(rollout.value_returns[0])

    sink._publish_value_rollouts([rollout])

    value_sample = publisher.publish.call_args.args[0].samples[0]
    assert value_sample.token_ids == [1, 90, 91, 2]
    assert value_sample.mask == [False, False, False, True]
    assert value_sample.targets == [0.0, 0.0, 0.0, 1.0]
    assert rollout.samples[0].token_ids == policy_token_ids
    assert rollout.samples[0].mask == policy_mask
    assert rollout.value_returns[0] == policy_targets


def test_value_prefix_preflight_rejects_overflow_without_exposing_prompt():
    sink = TrainSink.__new__(TrainSink)
    sink._value_seq_len = 3
    sink.renderer = MagicMock()
    sink.renderer.render_ids.return_value = [1, 90, 91]
    sink.tokenizer = SimpleNamespace(bos_token_id=1)
    rollout = _rollout()
    rollout.task = SimpleNamespace(idx=7, value_function_prompt="private solved grid")

    with pytest.raises(ValueError, match="merged_tokens=4") as error:
        sink._build_value_prefix(rollout, rollout.samples, enabled=True)

    assert "private solved grid" not in str(error.value)
    sink.renderer.render_ids.assert_called_once_with(
        [{"role": "system", "content": "private solved grid"}],
        add_generation_prompt=False,
    )

    sink._value_seq_len = 4
    assert sink._build_value_prefix(rollout, rollout.samples, enabled=True) == TokenPrefix(
        token_ids=(90, 91),
        insert_at=1,
    )


def test_missing_or_none_value_prompt_uses_no_prefix_without_rendering():
    sink = TrainSink.__new__(TrainSink)
    sink._value_seq_len = 8
    sink.renderer = MagicMock()
    sink.tokenizer = SimpleNamespace(bos_token_id=1)
    rollout = _rollout()

    assert sink._build_value_prefix(rollout, rollout.samples, enabled=True) is None
    rollout.task.value_function_prompt = None
    assert sink._build_value_prefix(rollout, rollout.samples, enabled=True) is None
    rollout.task.value_function_prompt = {"ignored": "without a value function"}
    assert sink._build_value_prefix(rollout, rollout.samples, enabled=False) is None
    sink.renderer.render_ids.assert_not_called()


def test_value_prefix_preflight_rejects_empty_prompt_and_mixed_bos_branches():
    sink = TrainSink.__new__(TrainSink)
    sink._value_seq_len = 8
    sink.renderer = MagicMock()
    sink.renderer.render_ids.return_value = [1, 90]
    sink.tokenizer = SimpleNamespace(bos_token_id=1)
    rollout = _rollout(num_samples=2)
    rollout.task = SimpleNamespace(idx=7, value_function_prompt="  ")

    with pytest.raises(ValueError, match="non-whitespace"):
        sink._build_value_prefix(rollout, rollout.samples, enabled=True)

    rollout.task.value_function_prompt = "hint"
    rollout.samples[1].token_ids[0] = 2
    with pytest.raises(ValueError, match="all branches"):
        sink._build_value_prefix(rollout, rollout.samples, enabled=True)


def test_value_prefix_preflight_rejects_non_string_and_rendered_bos_mismatch():
    sink = TrainSink.__new__(TrainSink)
    sink._value_seq_len = 8
    sink.renderer = MagicMock()
    sink.tokenizer = SimpleNamespace(bos_token_id=1)
    rollout = _rollout()
    rollout.task = SimpleNamespace(idx=7, value_function_prompt={"oracle": "private"})

    with pytest.raises(TypeError, match="must be a string"):
        sink._build_value_prefix(rollout, rollout.samples, enabled=True)
    sink.renderer.render_ids.assert_not_called()

    rollout.task.value_function_prompt = "hint"
    sink.renderer.render_ids.return_value = [90, 91]
    with pytest.raises(ValueError, match="same BOS convention"):
        sink._build_value_prefix(rollout, rollout.samples, enabled=True)


def test_overflow_escapes_add_before_rollout_or_pending_state_mutates():
    async def run_test() -> None:
        old_samples = [_sample()]
        old_prefix = TokenPrefix(token_ids=(70,), insert_at=1)
        overflow_sample = _sample()
        overflow_sample.token_ids = [1, 2, 3]
        overflow_sample.mask = [False, True, True]
        overflow_sample.logprobs = [0.0, -0.1, -0.2]
        rollout = _rollout()
        rollout.group_id = uuid.UUID(int=1)
        rollout.task = SimpleNamespace(idx=7, value_function_prompt="private oracle")
        rollout.samples = old_samples
        rollout.value_prefix = old_prefix

        algorithm = SimpleNamespace(
            value_evaluator=SimpleNamespace(evaluate=AsyncMock()),
            finalize_rollout=AsyncMock(),
        )
        sink = TrainSink.__new__(TrainSink)
        sink.mm_token_type_ids_mapping = None
        sink.train_envs = SimpleNamespace(get=lambda _name: SimpleNamespace(algorithm=algorithm))
        sink._value_seq_len = 3
        sink.renderer = MagicMock()
        sink.renderer.render_ids.return_value = [1, 90]
        sink.tokenizer = SimpleNamespace(bos_token_id=1)
        sink.scoring_tasks = {}
        sink.pending_rollouts = []
        sink.pending_groups = defaultdict(list)
        sink.pending_batch = []
        sink._value_publisher = MagicMock()

        with (
            patch(
                "prime_rl.orchestrator.train_sink.trace_to_samples",
                return_value=[overflow_sample],
            ),
            pytest.raises(ValueError, match="conditioned value input exceeds"),
        ):
            await sink.add(rollout)

        assert rollout.samples is old_samples
        assert rollout.value_prefix is old_prefix
        assert sink.scoring_tasks == {}
        assert sink.pending_rollouts == []
        assert sink.pending_groups == {}
        assert sink.pending_batch == []
        algorithm.finalize_rollout.assert_not_awaited()
        algorithm.value_evaluator.evaluate.assert_not_awaited()
        sink._value_publisher.publish.assert_not_called()

    asyncio.run(run_test())


def test_value_rollouts_are_ignored_when_all_targets_are_absent():
    publisher = MagicMock()
    sink = _value_sink(publisher)
    rollout = _rollout()
    rollout.value_returns = None

    sink._publish_value_rollouts([_rollout(with_sample=False), rollout])

    publisher.publish.assert_not_called()


def test_value_rollout_is_published_after_group_finalization():
    async def run_test() -> None:
        publisher = MagicMock()
        rollout = _rollout()
        rollout.value_returns = None
        rollout.value_version = None
        sink, env = _group_sink([rollout], minimum_group_size=1, value_publisher=publisher)

        async def attach_targets(survivors) -> None:
            survivors[0].value_returns = [[0.0, 1.0]]
            survivors[0].value_version = 7

        env.algorithm.finalize_group.side_effect = attach_targets

        await sink.process_group(uuid.UUID(int=0))

        published = publisher.publish.call_args.args[0]
        assert published.value_version == 7
        assert sink.pending_batch == [rollout]

    asyncio.run(run_test())


def test_process_batch_tracks_actual_shipped_value_versions():
    version_two = _rollout(value_version=2)
    version_three = _rollout(value_version=3)
    filtered_stale = _rollout(value_version=0)
    filtered_stale.is_filtered = True
    overflow_stale = _rollout(value_version=0)
    arrival_only_stale = _rollout(value_version=0)

    sink = TrainSink.__new__(TrainSink)
    sink.batch_size = 3
    sink.token_batch_size = None
    sink.pending_batch = [version_two, version_three, filtered_stale, overflow_stale]
    sink.pending_tokens = 0
    sink.pending_rollouts = TrainRollouts([arrival_only_stale])
    sink.post_filters = []

    batch = sink.process_batch()

    assert batch.shipped_value_version_min == 2
    assert batch.samples == [*version_two.samples, *version_three.samples]
    assert sink.pending_batch == [overflow_stale]

    fresh = _rollout(value_version=1)
    sink.batch_size = 2
    sink.pending_batch = [overflow_stale, fresh]
    sink.pending_rollouts = TrainRollouts()

    assert sink.process_batch().shipped_value_version_min == 0


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
        await sink.stop()

        assert task.cancelled()
        assert sink.scoring_tasks == {}

    asyncio.run(run_test())
