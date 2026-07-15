"""Shared rollout-credit math for orchestrator algorithms."""

from __future__ import annotations

import math

from prime_rl.configs.algorithm import LinearMixBaselineConfig


def compute_gae(
    *,
    reward: float,
    values: list[float],
    mask: list[bool],
    gamma: float,
    gae_lambda: float,
    value_target_lambda: float,
) -> tuple[list[float], list[float]]:
    """Policy GAE and independently parameterized TD(lambda) value targets."""
    if len(values) != len(mask):
        raise ValueError(f"value/mask length mismatch: {len(values)} != {len(mask)}")
    action_indices = [index for index, trainable in enumerate(mask) if trainable]
    advantages = [0.0] * len(mask)
    returns = [0.0] * len(mask)
    next_policy_gae = 0.0
    next_target_advantage = 0.0
    for action_position in reversed(range(len(action_indices))):
        index = action_indices[action_position]
        has_next = action_position + 1 < len(action_indices)
        next_value = values[action_indices[action_position + 1]] if has_next else 0.0
        immediate_reward = reward if not has_next else 0.0
        nonterminal = 1.0 if has_next else 0.0
        delta = immediate_reward + gamma * next_value * nonterminal - values[index]
        next_policy_gae = delta + gamma * gae_lambda * nonterminal * next_policy_gae
        next_target_advantage = delta + gamma * value_target_lambda * nonterminal * next_target_advantage
        advantages[index] = next_policy_gae
        returns[index] = next_target_advantage + values[index]
    return advantages, returns


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


def linear_mix_advantages(
    group_advantage: float,
    value_advantages: list[float],
    mask: list[bool],
    config: LinearMixBaselineConfig,
) -> list[float]:
    if len(value_advantages) != len(mask):
        raise ValueError("value advantage/mask length mismatch")
    output = [0.0] * len(mask)
    for index, trainable in enumerate(mask):
        if not trainable:
            continue
        output[index] = (1.0 - config.rho) * group_advantage + config.rho * value_advantages[index]
    return output
