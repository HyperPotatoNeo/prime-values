from collections import deque

from prime_rl.value.coordinator import admit_available_rollouts
from prime_rl.value.replay import ValueReplayBuffer
from prime_rl.value.types import ValueTrainingRollout, ValueTrainingSample


def _rollout(rollout_id: int) -> ValueTrainingRollout:
    return ValueTrainingRollout(
        samples=[ValueTrainingSample(token_ids=[rollout_id], mask=[True], targets=[0.0])],
        rollout_id=rollout_id,
        policy_version=0,
        value_version=0,
    )


class _Receiver:
    def __init__(self, rollouts: list[ValueTrainingRollout]) -> None:
        self.rollouts = deque(rollouts)
        self.requests: list[tuple[int, bool]] = []

    def receive_available(self, limit: int, *, wait_for_first: bool = False) -> list[ValueTrainingRollout]:
        self.requests.append((limit, wait_for_first))
        return [self.rollouts.popleft() for _ in range(min(limit, len(self.rollouts)))]


def _replay() -> ValueReplayBuffer:
    return ValueReplayBuffer(
        batch_size=2,
        capacity=4,
        refill_size=4,
        max_updates_per_rollout=3,
    )


def test_ready_admission_drains_one_batch_and_preserves_capacity():
    replay = _replay()
    replay.extend(_rollout(rollout_id) for rollout_id in range(4))
    receiver = _Receiver([_rollout(4), _rollout(5), _rollout(6)])

    admitted = admit_available_rollouts(receiver, replay)

    assert admitted == 2
    assert receiver.requests == [(2, False)]
    assert len(receiver.rollouts) == 1
    assert len(replay) == replay.capacity
    assert replay.snapshot().evicted == 2


def test_filling_admission_stops_exactly_at_refill_threshold():
    replay = _replay()
    replay.extend([_rollout(0), _rollout(1), _rollout(2)])
    receiver = _Receiver([_rollout(3), _rollout(4)])

    admitted = admit_available_rollouts(receiver, replay, wait_for_first=True)

    assert admitted == 1
    assert receiver.requests == [(1, True)]
    assert len(receiver.rollouts) == 1
    assert replay.can_sample


def test_one_update_replay_streams_consecutive_cohorts_once_in_order():
    replay = ValueReplayBuffer(
        batch_size=2,
        capacity=2,
        refill_size=2,
        max_updates_per_rollout=1,
    )
    receiver = _Receiver([_rollout(0)])

    assert admit_available_rollouts(receiver, replay, wait_for_first=True) == 1
    assert not replay.can_sample

    receiver.rollouts.extend([_rollout(1), _rollout(2), _rollout(3)])
    assert admit_available_rollouts(receiver, replay, wait_for_first=True) == 1
    first = replay.sample()
    assert [sample.token_ids[0] for sample in first.samples] == [0, 1]
    assert len(replay) == 0

    assert admit_available_rollouts(receiver, replay, wait_for_first=True) == 2
    second = replay.sample()
    assert [sample.token_ids[0] for sample in second.samples] == [2, 3]
    assert len(replay) == 0
    assert replay.snapshot().retired == 4
    assert receiver.requests == [(2, True), (1, True), (2, True)]
