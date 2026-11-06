"""Token-aligned critic credit, independent of policy filtering and transport."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prime_rl.configs.value import ValueFunctionConfig
    from prime_rl.orchestrator.trajectories import TrainingLayout
    from prime_rl.orchestrator.value_context import TokenPrefix
    from prime_rl.value.trajectory import ValueTrajectory


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


def value_input(token_ids: Sequence[int], prefix: TokenPrefix | None, seq_len: int) -> list[int]:
    tokens = list(token_ids)
    return prefix.apply(tokens) if prefix is not None else tokens[:seq_len]


def sequential_value_layout(
    trajectory: ValueTrajectory,
    config: ValueFunctionConfig,
    *,
    uses_policy_gae: bool,
    branch_semantics: str | None,
) -> TrainingLayout | None:
    needs_stitching = (
        config.gamma < 1.0 or config.value_target_lambda < 1.0 or (uses_policy_gae and config.gae_lambda < 1.0)
    )
    if len(trajectory.samples) <= 1 or not needs_stitching:
        return None
    if branch_semantics != "sequential":
        raise ValueError(
            "temporal value credit over multiple branches requires "
            "algo.branch_semantics='sequential'; "
            f"env={trajectory.env_name!r}, group={trajectory.group_id}, "
            f"branches={len(trajectory.samples)}, gamma={config.gamma}, "
            f"gae_lambda={config.gae_lambda}, value_target_lambda={config.value_target_lambda}"
        )
    layout = trajectory.training_layout
    assert layout is not None
    assert config.model is not None
    if trajectory.prefix is not None:
        return layout
    visible_lengths = [min(len(sample.token_ids), config.model.seq_len) for sample in trajectory.samples]
    for span in layout.trainable_spans:
        end = span.sample_start + span.length
        visible_length = visible_lengths[span.sample_index]
        if end > visible_length:
            raise ValueError(
                "sequential temporal credit requires every trainable token to be visible "
                f"to the critic; sample {span.sample_index} span [{span.sample_start}, {end}) "
                f"exceeds visible length {visible_length}"
            )
    return layout


def compute_value_result(
    trajectory: ValueTrajectory,
    predictions: list[list[float]],
    config: ValueFunctionConfig,
    *,
    layout: TrainingLayout | None,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    assert config.model is not None
    if len(predictions) != len(trajectory.samples):
        raise ValueError(
            f"value evaluator returned {len(predictions)} branches for rollout with {len(trajectory.samples)} samples"
        )
    projected_predictions: list[list[float]] = []
    for sample, values in zip(trajectory.samples, predictions, strict=True):
        input_tokens = value_input(sample.token_ids, trajectory.prefix, config.model.seq_len)
        if len(values) != len(input_tokens):
            if trajectory.prefix is None:
                raise ValueError(
                    f"value evaluator returned {len(values)} tokens for truncated value input of {len(input_tokens)}"
                )
            raise ValueError(
                f"value evaluator returned {len(values)} tokens for conditioned value input of {len(input_tokens)}"
            )
        projected = trajectory.prefix.project(values) if trajectory.prefix is not None else values
        projected_predictions.append(projected)

    if layout is None:
        advantages: list[list[float]] = []
        returns: list[list[float]] = []
        for sample, projected in zip(trajectory.samples, projected_predictions, strict=True):
            value_length = len(projected)
            sample_advantages, sample_returns = compute_gae(
                reward=float(trajectory.reward),
                values=projected,
                mask=list(sample.mask[:value_length]),
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                value_target_lambda=config.value_target_lambda,
            )
            padding = len(sample.token_ids) - value_length
            advantages.append(sample_advantages + [0.0] * padding)
            returns.append(sample_returns + [0.0] * padding)
    else:
        flat_values = [
            value
            for span in layout.trainable_spans
            for value in projected_predictions[span.sample_index][span.sample_start : span.sample_start + span.length]
        ]
        flat_advantages, flat_returns = compute_gae(
            reward=float(trajectory.reward),
            values=flat_values,
            mask=[True] * len(flat_values),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            value_target_lambda=config.value_target_lambda,
        )
        advantages = [[0.0] * len(sample.token_ids) for sample in trajectory.samples]
        returns = [[0.0] * len(sample.token_ids) for sample in trajectory.samples]
        offset = 0
        for span in layout.trainable_spans:
            end = offset + span.length
            sample_end = span.sample_start + span.length
            advantages[span.sample_index][span.sample_start : sample_end] = flat_advantages[offset:end]
            returns[span.sample_index][span.sample_start : sample_end] = flat_returns[offset:end]
            offset = end
    value_predictions = [
        values + [0.0] * (len(sample.token_ids) - len(values))
        for sample, values in zip(trajectory.samples, projected_predictions, strict=True)
    ]
    return value_predictions, advantages, returns
