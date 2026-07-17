from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.advantage import group_advantages
from prime_rl.orchestrator.algo.base import Algorithm

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

    @property
    def minimum_group_size(self) -> int:
        return 2 if self.baseline.type == "leave_one_out" else 1

    async def score_group(self, group: list[Rollout]) -> None:
        raw_rewards = [float(rollout.reward) for rollout in group]
        rewards = torch.tensor(raw_rewards, dtype=torch.float32)
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

        raise TypeError(f"unsupported GRPO baseline {type(baseline).__name__}")
