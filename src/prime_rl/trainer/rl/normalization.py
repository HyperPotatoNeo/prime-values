"""Token-count normalization shared by the policy trainer and numerical probes."""

from collections.abc import Iterable, Mapping

import torch
import torch.distributed as dist


def count_loss_tokens(micro_batches: Iterable[Mapping[str, torch.Tensor | None]]) -> tuple[int, int, int]:
    rl_count = ce_count = ref_count = 0
    for batch in micro_batches:
        mask = batch["loss_mask"]
        assert mask is not None
        rl_weights = batch["rl_weights"]
        rl_count += int((mask & (rl_weights != 0)).sum()) if rl_weights is not None else int(mask.sum())
        if batch["ce_weights"] is not None:
            ce_count += int((batch["ce_weights"] != 0).sum())
        if batch["ref_kl_weights"] is not None:
            ref_count += int((batch["ref_kl_weights"] != 0).sum())
    return rl_count, ce_count, ref_count


def global_loss_scales(
    micro_batches: Iterable[Mapping[str, torch.Tensor | None]],
    *,
    device: torch.device | str,
    group: dist.ProcessGroup | None = None,
) -> tuple[int, int, int]:
    counts = torch.tensor(count_loss_tokens(micro_batches), dtype=torch.int64, device=device)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
    return tuple(max(count, 1) for count in counts.tolist())
