import random

import pytest

from prime_rl.value.replay import ValueReplayBuffer
from prime_rl.value.types import ValueTrainingRollout, ValueTrainingSample


def _rollout(
    rollout_id: int,
    *,
    branches: int = 1,
    trainable: bool = True,
) -> ValueTrainingRollout:
    samples = [
        ValueTrainingSample(
            token_ids=[rollout_id, branch],
            mask=[trainable, trainable],
            targets=[float(rollout_id), float(branch)],
        )
        for branch in range(branches)
    ]
    return ValueTrainingRollout(
        samples=samples,
        rollout_id=rollout_id,
        policy_version=rollout_id // 2,
        value_version=rollout_id // 3,
    )


def _buffer(
    *,
    batch_size: int = 2,
    capacity: int = 4,
    refill_size: int = 4,
    max_updates_per_rollout: int = 2,
    seed: int = 0,
) -> ValueReplayBuffer:
    return ValueReplayBuffer(
        batch_size=batch_size,
        capacity=capacity,
        refill_size=refill_size,
        max_updates_per_rollout=max_updates_per_rollout,
        seed=seed,
    )


def _selected_rollout_ids(batch) -> list[int]:
    return [sample.token_ids[0] for sample in batch.samples]


def test_one_update_replay_preserves_the_full_cohort_and_order():
    replay = _buffer(batch_size=3, capacity=3, refill_size=3, max_updates_per_rollout=1)
    replay.extend(_rollout(rollout_id) for rollout_id in [7, 8, 9])

    batch = replay.sample()

    assert _selected_rollout_ids(batch) == [7, 8, 9]
    assert batch.num_rollouts == 3
    assert batch.rollout_id_min == 7
    assert batch.rollout_id_max == 9
    assert batch.replay_attempt_min == batch.replay_attempt_max == 1
    assert batch.replay_attempt_mean == 1.0
    assert len(replay) == 0
    assert not replay.can_sample


def test_replay_refills_hysteretically_after_falling_below_one_batch():
    replay = _buffer(batch_size=2, capacity=4, refill_size=4, max_updates_per_rollout=1)
    replay.extend(_rollout(rollout_id) for rollout_id in range(4))
    assert replay.can_sample

    replay.sample()
    assert replay.can_sample
    replay.sample()
    assert not replay.can_sample

    replay.extend([_rollout(4), _rollout(5), _rollout(6)])
    assert len(replay) == 3
    assert not replay.can_sample
    replay.add(_rollout(7))
    assert replay.can_sample


def test_admission_limit_caps_each_refill_or_ready_turn_to_one_batch():
    replay = _buffer(batch_size=3, capacity=6, refill_size=5, max_updates_per_rollout=3)
    assert replay.admission_limit == 3

    replay.extend(_rollout(rollout_id) for rollout_id in range(3))
    assert replay.admission_limit == 2

    replay.extend([_rollout(3), _rollout(4)])
    assert replay.can_sample
    assert replay.admission_limit == 3


def test_replay_uniformly_selects_distinct_rollouts_then_restores_fifo_order():
    first = _buffer(batch_size=3, capacity=6, refill_size=6, max_updates_per_rollout=3, seed=19)
    second = _buffer(batch_size=3, capacity=6, refill_size=6, max_updates_per_rollout=3, seed=19)
    rollouts = [_rollout(rollout_id) for rollout_id in range(6)]
    first.extend(rollouts)
    second.extend(rollouts)

    first_ids = _selected_rollout_ids(first.sample())
    second_ids = _selected_rollout_ids(second.sample())

    assert first_ids == second_ids
    assert first_ids == sorted(first_ids)
    assert len(set(first_ids)) == 3


def test_fifo_admission_evicts_the_oldest_rollout_without_refreshing_sampled_age():
    replay = _buffer(batch_size=4, capacity=4, refill_size=4, max_updates_per_rollout=3, seed=0)
    replay.extend(_rollout(rollout_id) for rollout_id in range(4))
    assert _selected_rollout_ids(replay.sample()) == [0, 1, 2, 3]

    replay.add(_rollout(4))

    snapshot = replay.snapshot()
    assert snapshot.size == 4
    assert snapshot.evicted == 1
    assert snapshot.admitted == 5
    assert _selected_rollout_ids(replay.sample()) == [1, 2, 3, 4]


def test_rollouts_retire_at_the_attempt_cap():
    replay = _buffer(batch_size=2, capacity=2, refill_size=2, max_updates_per_rollout=2)
    replay.extend([_rollout(0), _rollout(1)])

    first = replay.sample()
    second = replay.sample()

    assert first.replay_attempt_mean == 1.0
    assert second.replay_attempt_mean == 2.0
    assert replay.snapshot().attempts == 4
    assert replay.snapshot().retired == 2
    assert len(replay) == 0


def test_multibranch_rollouts_are_selected_and_retired_atomically():
    replay = _buffer(batch_size=1, capacity=1, refill_size=1, max_updates_per_rollout=1)
    replay.add(_rollout(4, branches=3))

    batch = replay.sample()

    assert batch.num_rollouts == 1
    assert _selected_rollout_ids(batch) == [4, 4, 4]
    assert [sample.token_ids[1] for sample in batch.samples] == [0, 1, 2]
    assert replay.snapshot().samples == 0
    assert replay.snapshot().tokens == 0


def test_zero_mask_rollouts_still_consume_an_attempt():
    replay = _buffer(batch_size=1, capacity=1, refill_size=1, max_updates_per_rollout=1)
    replay.add(_rollout(2, trainable=False))

    batch = replay.sample()

    assert not any(batch.samples[0].mask)
    assert replay.snapshot().attempts == 1
    assert replay.snapshot().retired == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0, "capacity": 1, "refill_size": 1, "max_updates_per_rollout": 1},
        {"batch_size": 2, "capacity": 1, "refill_size": 1, "max_updates_per_rollout": 1},
        {"batch_size": 2, "capacity": 4, "refill_size": 1, "max_updates_per_rollout": 1},
        {"batch_size": 2, "capacity": 4, "refill_size": 5, "max_updates_per_rollout": 1},
        {"batch_size": 1, "capacity": 1, "refill_size": 1, "max_updates_per_rollout": 0},
    ],
)
def test_replay_rejects_invalid_shape(kwargs):
    with pytest.raises(ValueError):
        ValueReplayBuffer(**kwargs)


def test_replay_rejects_sampleless_rollouts_and_sampling_while_filling():
    replay = _buffer()
    empty = ValueTrainingRollout(samples=[], rollout_id=0, policy_version=0, value_version=0)

    with pytest.raises(ValueError, match="at least one sample"):
        replay.add(empty)
    with pytest.raises(RuntimeError, match="filling"):
        replay.sample()


def test_mixed_admission_and_sampling_preserves_replay_invariants():
    replay = _buffer(
        batch_size=4,
        capacity=12,
        refill_size=8,
        max_updates_per_rollout=3,
        seed=29,
    )
    operation_rng = random.Random(11)
    attempts: dict[int, int] = {}
    next_rollout_id = 0
    draws = 0

    for _ in range(200):
        if not replay.can_sample or operation_rng.random() < 0.6:
            replay.add(_rollout(next_rollout_id))
            next_rollout_id += 1
        else:
            selected = _selected_rollout_ids(replay.sample())
            assert len(selected) == len(set(selected)) == replay.batch_size
            for rollout_id in selected:
                attempts[rollout_id] = attempts.get(rollout_id, 0) + 1
                assert attempts[rollout_id] <= replay.max_updates_per_rollout
            draws += 1

        assert len(replay) <= replay.capacity
        assert not replay.can_sample or len(replay) >= replay.batch_size

    assert draws > 10
