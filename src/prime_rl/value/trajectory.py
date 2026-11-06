"""Value-owned inputs copied from the canonical policy materialization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prime_rl.orchestrator.trajectories import TrainingLayout
from prime_rl.orchestrator.value_context import TokenPrefix

if TYPE_CHECKING:
    from prime_rl.transport import TrainingSample


@dataclass(frozen=True)
class ValueTrajectorySample:
    token_ids: tuple[int, ...]
    mask: tuple[bool, ...]
    sampled_mask: tuple[bool, ...]


@dataclass(frozen=True)
class ValueTrajectory:
    samples: tuple[ValueTrajectorySample, ...]
    training_layout: TrainingLayout | None
    reward: float
    prefix: TokenPrefix | None
    env_name: str
    group_id: str
    policy_version: int


def snapshot_value_trajectory(
    samples: Sequence[TrainingSample],
    *,
    sampled_masks: Sequence[Sequence[bool]],
    layout: TrainingLayout | None,
    reward: float,
    prefix: TokenPrefix | None = None,
    env_name: str = "",
    group_id: str = "",
    policy_version: int = 0,
) -> ValueTrajectory:
    owned = []
    for sample, sampled_mask in zip(samples, sampled_masks, strict=True):
        if not len(sample.token_ids) == len(sample.mask) == len(sampled_mask):
            raise ValueError("value trajectory token and mask streams must align")
        if any(trainable and not sampled for trainable, sampled in zip(sample.mask, sampled_mask, strict=True)):
            raise ValueError("value trajectory train mask must follow native sampled provenance")
        owned.append(
            ValueTrajectorySample(
                token_ids=tuple(sample.token_ids),
                mask=tuple(sample.mask),
                sampled_mask=tuple(sampled_mask),
            )
        )
    return ValueTrajectory(
        samples=tuple(owned),
        training_layout=layout,
        reward=float(reward),
        prefix=prefix,
        env_name=env_name,
        group_id=str(group_id),
        policy_version=policy_version,
    )
