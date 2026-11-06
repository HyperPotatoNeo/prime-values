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


SAMPLED_AGE_BINS = (
    ("0", 0, 1),
    ("1", 1, 2),
    ("2-3", 2, 4),
    ("4-7", 4, 8),
    ("8-15", 8, 16),
    ("16-31", 16, 32),
    ("32-63", 32, 64),
    ("64+", 64, None),
)


def policy_version_reference(*, step: int, max_concurrent_runs: int, has_lora: bool) -> int | None:
    """Only base-model, single-run steps identify the weights used by this forward."""
    if max_concurrent_runs != 1 or has_lora:
        return None
    if type(step) is not int or not 1 <= step <= 2**63:
        raise ValueError("A policy step must identify an int64 pre-update weight version")
    return step - 1


@torch.no_grad()
def accumulate_policy_provenance(
    summaries: TensorSummaries,
    *,
    token_policy_versions: Tensor | None,
    eligible_mask: Tensor,
    reference_version: int | None,
    observe: bool = True,
    mismatch_kl: Tensor | None = None,
) -> None:
    """Observe the validated physical-axis carrier, without changing token eligibility.

    CP forwards gather full sequences on every CP rank. Only CP rank zero owns
    these observations; every rank still participates in compute_stats below.
    """
    if not observe:
        return
    if eligible_mask.dtype != torch.bool:
        raise ValueError("Policy provenance eligibility must be boolean")
    if token_policy_versions is not None:
        if token_policy_versions.dtype != torch.int64 or token_policy_versions.shape != eligible_mask.shape:
            raise ValueError("Policy versions must be int64 on the full physical token axis")
        versions = token_policy_versions.detach().to(eligible_mask.device, non_blocking=True)
        known = versions >= 0
    else:
        versions = None
        known = torch.zeros_like(eligible_mask)
    if mismatch_kl is not None:
        if mismatch_kl.shape != eligible_mask.shape or mismatch_kl.device != eligible_mask.device:
            raise ValueError("Mismatch values must match the physical token mask shape and device")
        mismatch_kl = mismatch_kl.detach()
        saturated = mismatch_kl >= DIAGNOSTIC_RATIO_LIMIT
        nonfinite_nan = mismatch_kl.isnan()

    def add_mismatch(population: str, mask: Tensor) -> None:
        if mismatch_kl is None:
            return
        key = f"mismatch_kl_by_sampled_age/{population}"
        summaries.add(key, mismatch_kl, mask)
        summaries.add(f"{key}/saturated", saturated, mask)
        # Distributed MIN/MAX need not propagate a NaN on every backend.
        summaries.add(f"{key}/nan", nonfinite_nan, mask)

    one = torch.ones_like(eligible_mask)
    summaries.add("observed_policy_version/eligible", one, eligible_mask)
    summaries.add("observed_policy_version/known", one, eligible_mask & known)
    summaries.add("observed_policy_version/unknown", one, eligible_mask & ~known)
    add_mismatch("unknown", eligible_mask & ~known)
    if reference_version is None:
        return
    if type(reference_version) is not int or not 0 <= reference_version <= 2**63 - 1:
        raise ValueError("The reference policy version must be a nonnegative int64")
    future = versions > reference_version if versions is not None else known
    valid_age = eligible_mask & known & ~future
    summaries.add("observed_policy_version/future", one, eligible_mask & future)
    # Replace unknown/future labels before subtraction: INT64_MAX and -1 must
    # never overflow or become invented negative/zero ages in the population.
    ages = (
        reference_version - torch.where(valid_age, versions, reference_version)
        if versions is not None
        else torch.zeros_like(eligible_mask, dtype=torch.int64)
    )
    summaries.add("sampled_weight_age", ages, valid_age)
    add_mismatch("future", eligible_mask & future)
    for name, lower, upper in SAMPLED_AGE_BINS:
        member = valid_age & (ages >= lower)
        if upper is not None:
            member = member & (ages < upper)
        add_mismatch(name, member)


def compute_policy_provenance(summaries: TensorSummaries) -> dict[str, float | int]:
    reduced = summaries.compute_stats()
    if not reduced:
        return {}
    prefix = "observed_policy_version/"
    eligible = reduced[f"{prefix}eligible/count"]
    metrics = {f"{prefix}eligible_count": eligible}
    for population in ("known", "unknown", "future"):
        key = f"{prefix}{population}"
        if f"{key}/count" in reduced:
            count = reduced[f"{key}/count"]
            metrics[f"{key}_count"] = count
            if eligible:
                metrics[f"{key}_fraction"] = count / eligible
    if "sampled_weight_age/count" in reduced:
        count = reduced["sampled_weight_age/count"]
        metrics["sampled_weight_age/count"] = count
        if count:
            for stat in ("mean", "max"):
                metrics[f"sampled_weight_age/{stat}"] = reduced[f"sampled_weight_age/{stat}"]
    for population in [*(name for name, _, _ in SAMPLED_AGE_BINS), "unknown", "future"]:
        key = f"mismatch_kl_by_sampled_age/{population}"
        if f"{key}/count" not in reduced:
            continue
        count = reduced[f"{key}/count"]
        metrics[f"{key}/count"] = count
        if count:
            has_nan = reduced[f"{key}/nan/mean"] > 0
            for stat in ("mean", "max"):
                metrics[f"{key}/{stat}"] = float("nan") if has_nan else reduced[f"{key}/{stat}"]
            metrics[f"{key}/saturated_fraction"] = reduced[f"{key}/saturated/mean"]
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
