"""Shared rollout-credit math for orchestrator algorithms."""

from __future__ import annotations

import math

from prime_rl.value.credit import compute_gae as compute_gae


def group_baselines(rewards: list[float], baseline: str) -> list[float]:
    if not rewards:
        return []
    if baseline == "mean":
        mean = math.fsum(rewards) / len(rewards)
        return [mean] * len(rewards)
    if baseline == "leave_one_out":
        if len(rewards) < 2:
            raise ValueError("leave_one_out baseline requires group_size >= 2")
        # Sum siblings directly rather than reconstructing their mean from a
        # total that contains the rollout's own reward. Besides better numeric
        # behavior, this keeps the implemented LOO anchor literally free of
        # that reward.
        return [
            math.fsum(other for other_index, other in enumerate(rewards) if other_index != index) / (len(rewards) - 1)
            for index in range(len(rewards))
        ]
    raise ValueError(f"unsupported group baseline {baseline!r}")


def group_advantages(rewards: list[float], baseline: str) -> list[float]:
    return [
        reward - group_baseline
        for reward, group_baseline in zip(rewards, group_baselines(rewards, baseline), strict=True)
    ]
