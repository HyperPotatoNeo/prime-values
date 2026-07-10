from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.algo.tether import AdaptiveTetherCoefficients, tether_regression_stats
from prime_rl.value.math import group_advantages, group_baselines, linear_mix_advantages, tether_advantages

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
        policy_seq_len: int | None = None,
    ):
        super().__init__(
            config,
            policy_pool,
            value_evaluator=value_evaluator,
            value_config=value_config,
            policy_seq_len=policy_seq_len,
        )
        self.length_penalty = config.length_penalty
        self.baseline = config.baseline
        self.adaptive_tether: AdaptiveTetherCoefficients | None = None
        if self.baseline.type == "tether" and self.baseline.adaptive is not None:
            if value_config is None or value_config.batch_size is None:
                raise ValueError("adaptive tether needs a resolved value_function.batch_size")
            adaptive_batch_size = self.baseline.adaptive.batch_size or value_config.batch_size
            self.adaptive_tether = AdaptiveTetherCoefficients(
                self.baseline.adaptive,
                batch_size=adaptive_batch_size,
                reward_range=self.baseline.reward_range,
            )

    def metrics(self) -> dict[str, float]:
        return self.adaptive_tether.metrics() if self.adaptive_tether is not None else {}

    def metric_keys(self) -> list[str]:
        return self.adaptive_tether.metric_keys() if self.adaptive_tether is not None else []

    def state_dict(self) -> dict[str, Any]:
        if self.adaptive_tether is None:
            return {}
        return {"adaptive_tether": self.adaptive_tether.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        adaptive_state = state.get("adaptive_tether")
        if adaptive_state is None:
            return
        if self.adaptive_tether is None:
            raise ValueError("checkpoint contains adaptive TETHER state but adaptive mode is disabled")
        self.adaptive_tether.load_state_dict(adaptive_state)

    @property
    def minimum_group_size(self) -> int:
        group_baseline = getattr(self.baseline, "group", self.baseline.type)
        return 2 if group_baseline == "leave_one_out" else 1

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

        if baseline.type == "linear_mix":
            group_credit = group_advantages(raw_rewards, baseline.group)
            for rollout, scalar_group_advantage in zip(group, group_credit, strict=True):
                assert rollout.value_advantages is not None
                mixed: list[float] = []
                for sample, value_advantages in zip(rollout.samples, rollout.value_advantages, strict=True):
                    mixed.extend(linear_mix_advantages(scalar_group_advantage, value_advantages, sample.mask, baseline))
                rollout.assign_advantages(mixed)
            return

        if baseline.type == "tether":
            group_anchors = group_baselines(raw_rewards, baseline.group)
            alpha, rho = (
                self.adaptive_tether.coefficients
                if self.adaptive_tether is not None
                else (baseline.alpha, baseline.rho)
            )
            regression_stats = []
            for rollout, group_anchor in zip(group, group_anchors, strict=True):
                assert rollout.value_predictions is not None
                assert self.value_config is not None and self.value_config.model is not None
                corrected: list[float] = []
                regression_values: list[list[float]] = []
                regression_masks: list[list[bool]] = []
                for sample, values in zip(rollout.samples, rollout.value_predictions, strict=True):
                    value_length = min(
                        len(sample.token_ids),
                        self.value_config.model.seq_len,
                        self.policy_seq_len or len(sample.token_ids),
                    )
                    valid_values = values[:value_length]
                    valid_mask = sample.mask[:value_length]
                    branch_advantages = tether_advantages(
                        reward=float(rollout.reward),
                        group_anchor=group_anchor,
                        values=valid_values,
                        mask=valid_mask,
                        alpha=alpha,
                        rho=rho,
                        reward_range=baseline.reward_range,
                    )
                    corrected.extend(branch_advantages + [0.0] * (len(sample.token_ids) - value_length))
                    regression_values.append(valid_values)
                    regression_masks.append(valid_mask)
                rollout.assign_advantages(corrected)
                if self.adaptive_tether is not None:
                    regression_stats.append(
                        tether_regression_stats(
                            reward=float(rollout.reward),
                            group_anchor=group_anchor,
                            value_predictions=regression_values,
                            masks=regression_masks,
                            reward_range=baseline.reward_range,
                            alpha=alpha,
                            rho=rho,
                        )
                    )
            # Never let a reward tune the baseline used for its own group. The
            # entire group has already been assigned before these moments can
            # update coefficients for a later group.
            if self.adaptive_tether is not None:
                self.adaptive_tether.observe_group(regression_stats)
            return

        raise TypeError(f"unsupported GRPO baseline {type(baseline).__name__}")
