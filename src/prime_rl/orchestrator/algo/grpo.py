from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.value.math import group_advantages, linear_mix_advantages, tether_advantages

if TYPE_CHECKING:
    from prime_rl.configs.value import ValueFunctionConfig
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool
    from prime_rl.value.client import ValueEvaluatorClient


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; assign configured group/value credit; action tokens
    feed the ``rl`` loss."""

    def __init__(
        self,
        config: GRPOAlgoConfig,
        policy_pool: InferencePool,
        *,
        value_evaluator: ValueEvaluatorClient | None = None,
        value_config: ValueFunctionConfig | None = None,
    ):
        super().__init__(
            config,
            policy_pool,
            value_evaluator=value_evaluator,
            value_config=value_config,
        )
        self.length_penalty = config.length_penalty
        self.baseline = config.baseline

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        length_penalty = self.length_penalty
        shaped_rewards = rewards
        if length_penalty is not None:
            output = torch.tensor([rollout.num_output_tokens for rollout in group], dtype=rewards.dtype)
            total = torch.tensor([rollout.num_total_tokens for rollout in group], dtype=rewards.dtype)
            turns = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty

        baseline = self.baseline
        if baseline.type in {"mean", "leave_one_out"}:
            advantages = group_advantages(shaped_rewards.tolist(), baseline.type)
            for rollout, advantage in zip(group, advantages, strict=True):
                rollout.assign_advantages(advantage)
            return

        if baseline.type == "value":
            for rollout in group:
                if rollout.value_advantages is None:
                    raise RuntimeError("value evaluator did not attach GAE advantages")
                rollout.assign_advantages([value for branch in rollout.value_advantages for value in branch])
            return

        group_credit = group_advantages(rewards.tolist(), baseline.group)
        if baseline.type == "linear_mix":
            for rollout, scalar_group_advantage in zip(group, group_credit, strict=True):
                assert rollout.value_advantages is not None
                mixed: list[float] = []
                for sample, value_advantages in zip(rollout.samples, rollout.value_advantages, strict=True):
                    mixed.extend(linear_mix_advantages(scalar_group_advantage, value_advantages, sample.mask, baseline))
                rollout.assign_advantages(mixed)
            return

        if baseline.type == "tether":
            for rollout, scalar_group_advantage in zip(group, group_credit, strict=True):
                assert rollout.value_predictions is not None
                corrected: list[float] = []
                for sample, values in zip(rollout.samples, rollout.value_predictions, strict=True):
                    corrected.extend(
                        tether_advantages(
                            reward=float(rollout.reward),
                            group_advantage=scalar_group_advantage,
                            values=values,
                            mask=sample.mask,
                            config=baseline,
                        )
                    )
                rollout.assign_advantages(corrected)
            return

        raise TypeError(f"unsupported GRPO baseline {type(baseline).__name__}")
