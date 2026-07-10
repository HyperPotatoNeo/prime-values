from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from prime_rl.configs.algorithm import LinearMixBaselineConfig
from prime_rl.configs.value import ClassificationValueLossConfig, MSEValueLossConfig, ValueLossConfig


def value_head_output_size(config: ValueLossConfig) -> int:
    return config.num_bins if isinstance(config, ClassificationValueLossConfig) else 1


def align_value_logits(
    logits: Float[Tensor, "batch seq output"],
    sequence_lengths: list[int],
) -> Float[Tensor, "batch seq output"]:
    """Read action ``t`` from state ``t-1`` without crossing packed boundaries."""
    output_size = logits.shape[-1]
    flat = logits.reshape(-1, output_size)
    if sum(sequence_lengths) != flat.shape[0]:
        raise ValueError(f"sequence_lengths sum {sum(sequence_lengths)} does not match packed length {flat.shape[0]}")
    # Keep a zero-gradient path even when every packed sequence has length one
    # (the padding-only microbatch used to equalize work across DP ranks).
    # ``zeros_like`` would detach that rank from the graph and make its
    # collective backward fail before other ranks can reduce gradients.
    aligned = flat * 0.0
    offset = 0
    for length in sequence_lengths:
        if length > 1:
            aligned[offset + 1 : offset + length] = flat[offset : offset + length - 1]
        offset += length
    return aligned.reshape_as(logits)


def _support(config: ClassificationValueLossConfig, device: torch.device) -> Tensor:
    low, high = config.reward_range
    return torch.linspace(low, high, config.num_bins, dtype=torch.float32, device=device)


def predict_values(
    logits: Float[Tensor, "batch seq output"],
    config: ValueLossConfig,
) -> Float[Tensor, "batch seq"]:
    if isinstance(config, ClassificationValueLossConfig):
        return logits.float().softmax(dim=-1) @ _support(config, logits.device)
    return logits.squeeze(-1).float()


def _classification_target_distribution(targets: Tensor, config: ClassificationValueLossConfig) -> Tensor:
    """Project scalar targets onto adjacent support bins without changing their expectation."""
    low, high = config.reward_range
    tolerance = 1e-5 * max(high - low, 1.0)
    invalid = (targets < low - tolerance) | (targets > high + tolerance)
    if bool(invalid.any()):
        raise ValueError(
            f"classification value target {targets[invalid][0].item()} is outside reward_range={config.reward_range}"
        )
    normalized = (targets.float().clamp(low, high) - low) / (high - low)
    positions = normalized * (config.num_bins - 1)
    lower = positions.floor().long()
    upper = positions.ceil().long()
    upper_weight = positions - lower
    lower_weight = 1.0 - upper_weight
    distribution = torch.zeros(
        (*targets.shape, config.num_bins),
        dtype=torch.float32,
        device=targets.device,
    )
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


def compute_value_loss(
    logits: Float[Tensor, "batch seq output"],
    targets: Float[Tensor, "batch seq"],
    mask: Bool[Tensor, "batch seq"],
    config: ValueLossConfig,
    scale: int,
) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
    predictions = predict_values(logits, config)
    if not bool(mask.any()):
        empty = logits.new_empty(0)
        return logits.sum() * 0.0, {
            "value/loss": empty,
            "value/prediction": empty,
            "value/target": empty,
            "value/error": empty,
            "value/abs_error": empty,
            "value/squared_error": empty,
        }

    if isinstance(config, ClassificationValueLossConfig):
        target_distribution = _classification_target_distribution(targets[mask], config)
        log_probs = F.log_softmax(logits[mask].float(), dim=-1)
        probabilities = log_probs.exp()
        per_token = -(target_distribution * log_probs).sum(dim=-1)
        nearest_bin = target_distribution.argmax(dim=-1)
        accuracy = (logits[mask].argmax(dim=-1) == nearest_bin).float().detach()
        entropy = -(probabilities * log_probs).sum(dim=-1).detach()
        confidence = probabilities.max(dim=-1).values.detach()
    elif isinstance(config, MSEValueLossConfig):
        per_token = F.mse_loss(predictions[mask], targets[mask].float(), reduction="none")
        accuracy = None
        entropy = None
        confidence = None
    else:
        raise TypeError(f"unsupported value loss {type(config).__name__}")

    errors = predictions[mask] - targets[mask].float()
    metrics = {
        "value/loss": per_token.detach(),
        "value/prediction": predictions[mask].detach(),
        "value/target": targets[mask].detach(),
        "value/error": errors.detach(),
        "value/abs_error": errors.abs().detach(),
        "value/squared_error": errors.square().detach(),
    }
    if accuracy is not None:
        metrics["value/accuracy"] = accuracy
    if entropy is not None and confidence is not None:
        metrics["value/entropy"] = entropy
        metrics["value/confidence"] = confidence
    return per_token.sum() / max(scale, 1), metrics


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
            math.fsum(other for other_index, other in enumerate(rewards) if other_index != index)
            / (len(rewards) - 1)
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
