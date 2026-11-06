from __future__ import annotations

import torch
from torch import Tensor

from prime_rl.trainer.rl.loss import DIAGNOSTIC_RATIO_LIMIT, compute_mismatch_kl
from prime_rl.trainer.utils import TensorSummaries


def prepare_token_env_ids(env_names: list[str], device: torch.device) -> tuple[list[str], Tensor]:
    names = sorted(set(env_names))
    indices = {name: index for index, name in enumerate(names)}
    ids = torch.tensor([indices[name] for name in env_names], dtype=torch.int32, device="cpu")
    return names, ids.to(device, non_blocking=True)


@torch.no_grad()
def accumulate_token_metrics(
    summaries: TensorSummaries,
    diagnostics: TensorSummaries,
    *,
    entropy: Tensor,
    trainer_logprobs: Tensor,
    inference_logprobs: Tensor,
    loss_mask: Tensor,
    rl_weights: Tensor | None,
    ref_kl_weights: Tensor | None,
    env_names: list[str],
    env_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    if env_ids.numel() != loss_mask.numel():
        raise ValueError("Environment IDs must describe each token, including padding")
    mismatch_mask = loss_mask
    if rl_weights is not None or ref_kl_weights is not None:
        sampled_mask = (rl_weights != 0) if rl_weights is not None else loss_mask
        if ref_kl_weights is not None:
            sampled_mask = sampled_mask | (ref_kl_weights != 0)
        mismatch_mask = loss_mask & sampled_mask
    # Excluded NaN/Inf values must not participate even in the subtraction.
    log_ratio = torch.where(mismatch_mask, trainer_logprobs, 0.0) - torch.where(mismatch_mask, inference_logprobs, 0.0)
    mismatch_kl = compute_mismatch_kl(log_ratio)
    summaries.add("entropy/all", entropy, loss_mask)
    summaries.add("mismatch_kl/all", mismatch_kl, mismatch_mask)
    diagnostics.add("mismatch_kl/all/saturated", mismatch_kl >= DIAGNOSTIC_RATIO_LIMIT, mismatch_mask)
    for index, name in enumerate(env_names):
        member = env_ids == index
        summaries.add(f"entropy/{name}", entropy, loss_mask.reshape(-1) & member)
        summaries.add(f"mismatch_kl/{name}", mismatch_kl, mismatch_mask.reshape(-1) & member)
    return mismatch_kl, mismatch_mask


def compute_token_metrics(summaries: TensorSummaries, diagnostics: TensorSummaries) -> dict[str, float | int]:
    reduced = summaries.compute_stats()
    metrics = {
        key: value
        for key, value in reduced.items()
        if not key.endswith("/count")
        and (key.rsplit("/", 1)[0] == "entropy/all" or reduced[f"{key.rsplit('/', 1)[0]}/count"] > 0)
    }
    diagnostic_stats = diagnostics.compute_stats()
    # Unlike explicitly added loss diagnostic populations, this global stream
    # was absent when no sampling-logprob observation existed on any rank.
    if diagnostic_stats.get("mismatch_kl/all/saturated/count") == 0:
        for stat in ("count", "mean", "min", "max"):
            diagnostic_stats.pop(f"mismatch_kl/all/saturated/{stat}")
    metrics.update(diagnostic_stats)
    return metrics


def micro_step_message(
    micro_step: int,
    num_micro_batches: int,
    loss: Tensor,
    entropy: Tensor,
    loss_mask: Tensor,
    mismatch_kl: Tensor,
    mismatch_mask: Tensor,
    moe_stats: dict[str, Tensor | None],
) -> str:
    message = (
        f"Micro Step {micro_step}/{num_micro_batches} | Loss {loss.detach().mean().item():.4f}"
        f" | Entropy {entropy[loss_mask].detach().mean().item():.4f}"
    )
    if bool(mismatch_mask.any()):
        message += f" | Mismatch KL {mismatch_kl[mismatch_mask].mean().item():.4f}"
    for key, label in (("max_vio", "Max Vio"), ("routing_confidence", "Routing Conf.")):
        if (value := moe_stats.get(key)) is not None:
            message += f" | {label} {value.detach().mean().item():.4f}"
    return message
