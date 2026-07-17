from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from prime_rl.value.types import ValueTrainingBatch, ValueTrainingRollout


@dataclass(frozen=True)
class ValueReplaySnapshot:
    size: int
    samples: int
    tokens: int
    capacity: int
    refill_size: int
    ready: bool
    admitted: int
    attempts: int
    retired: int
    evicted: int


@dataclass
class _ReplayEntry:
    rollout: ValueTrainingRollout
    attempts: int = 0

    @property
    def samples(self) -> int:
        return len(self.rollout.samples)

    @property
    def tokens(self) -> int:
        return sum(len(sample.token_ids) for sample in self.rollout.samples)


class ValueReplayBuffer:
    """FIFO-admitted replay with uniform rollout sampling and hysteretic refill."""

    def __init__(
        self,
        *,
        batch_size: int,
        capacity: int,
        refill_size: int,
        max_updates_per_rollout: int,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if capacity < batch_size:
            raise ValueError("capacity must be at least batch_size")
        if not batch_size <= refill_size <= capacity:
            raise ValueError("refill_size must be between batch_size and capacity")
        if max_updates_per_rollout < 1:
            raise ValueError("max_updates_per_rollout must be positive")

        self.batch_size = batch_size
        self.capacity = capacity
        self.refill_size = refill_size
        self.max_updates_per_rollout = max_updates_per_rollout
        self._rng = random.Random(seed)
        self._entries: OrderedDict[int, _ReplayEntry] = OrderedDict()
        self._next_entry_id = 0
        self._ready = False
        self._samples = 0
        self._tokens = 0
        self._admitted = 0
        self._attempts = 0
        self._retired = 0
        self._evicted = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def can_sample(self) -> bool:
        return self._ready and len(self) >= self.batch_size

    @property
    def admission_limit(self) -> int:
        """Maximum rollouts to admit in the next bounded coordinator turn."""
        if self.can_sample:
            return self.batch_size
        return min(self.batch_size, max(self.refill_size - len(self), 0))

    def add(self, rollout: ValueTrainingRollout) -> None:
        if not rollout.samples:
            raise ValueError("value replay rollouts must contain at least one sample")
        if len(self) == self.capacity:
            _, evicted = self._entries.popitem(last=False)
            self._remove_totals(evicted)
            self._evicted += 1

        entry = _ReplayEntry(rollout=rollout)
        self._entries[self._next_entry_id] = entry
        self._next_entry_id += 1
        self._samples += entry.samples
        self._tokens += entry.tokens
        self._admitted += 1
        if not self._ready and len(self) >= self.refill_size:
            self._ready = True

    def extend(self, rollouts: Iterable[ValueTrainingRollout]) -> None:
        for rollout in rollouts:
            self.add(rollout)

    def sample(self) -> ValueTrainingBatch:
        if not self.can_sample:
            raise RuntimeError("value replay buffer is filling")

        entry_ids = self._rng.sample(tuple(self._entries), self.batch_size)
        entry_ids.sort()
        selected = [self._entries[entry_id] for entry_id in entry_ids]
        for entry in selected:
            entry.attempts += 1
        self._attempts += len(selected)

        attempts = [entry.attempts for entry in selected]
        rollouts = [entry.rollout for entry in selected]
        samples = [sample for rollout in rollouts for sample in rollout.samples]
        batch = ValueTrainingBatch(
            samples=samples,
            num_rollouts=len(rollouts),
            rollout_id_min=min(rollout.rollout_id for rollout in rollouts),
            rollout_id_max=max(rollout.rollout_id for rollout in rollouts),
            policy_version_min=min(rollout.policy_version for rollout in rollouts),
            policy_version_max=max(rollout.policy_version for rollout in rollouts),
            value_version_min=min(rollout.value_version for rollout in rollouts),
            value_version_max=max(rollout.value_version for rollout in rollouts),
            replay_attempt_min=min(attempts),
            replay_attempt_max=max(attempts),
            replay_attempt_mean=sum(attempts) / len(attempts),
        )

        for entry_id, entry in zip(entry_ids, selected, strict=True):
            if entry.attempts == self.max_updates_per_rollout:
                del self._entries[entry_id]
                self._remove_totals(entry)
                self._retired += 1
        if len(self) < self.batch_size:
            self._ready = False
        return batch

    def snapshot(self) -> ValueReplaySnapshot:
        return ValueReplaySnapshot(
            size=len(self),
            samples=self._samples,
            tokens=self._tokens,
            capacity=self.capacity,
            refill_size=self.refill_size,
            ready=self._ready,
            admitted=self._admitted,
            attempts=self._attempts,
            retired=self._retired,
            evicted=self._evicted,
        )

    def _remove_totals(self, entry: _ReplayEntry) -> None:
        self._samples -= entry.samples
        self._tokens -= entry.tokens
