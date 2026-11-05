"""Check the trainer's real count/reduction path against a probability oracle."""

import json
import math

import torch
import torch.distributed as dist

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn
from prime_rl.trainer.rl.normalization import global_loss_scales
from prime_rl.trainer.utils import Tensors, TensorSummaries, setup_torch_distributed
from prime_rl.trainer.world import get_world


def main():
    world = get_world()
    assert world.world_size in (1, 2)
    setup_torch_distributed()
    device = torch.device("cuda", world.local_rank)
    selected = slice(None) if world.world_size == 1 else slice(2 * world.rank, 2 * world.rank + 2)
    tensor = lambda values: torch.tensor(values, dtype=torch.float64, device=device)[selected]
    weights = tensor([0.5, 1.25, 2.0, 0.0])
    advantages = tensor([1.0, -2.0, 3.0, 0.0])
    mask = tensor([1, 1, 1, 0]).bool()
    ce_weights = tensor([0.0, 0.0, 0.0, 2.0])
    batch = {"loss_mask": mask, "rl_weights": weights, "ce_weights": ce_weights, "ref_kl_weights": None}
    scales = global_loss_scales([batch], device=device)
    assert scales == (3, 1, 1), scales
    theta = torch.tensor(math.log(0.4 / 0.6), dtype=torch.float64, device=device, requires_grad=True)
    logprobs = theta.sigmoid().log().expand_as(weights)
    loss, _ = compute_loss(
        trainer_logprobs=[logprobs],
        inference_logprobs=[torch.full_like(weights, math.log(0.3))],
        ref_logprobs=None,
        advantages=[advantages],
        loss_mask=[mask],
        rl_weights=[weights],
        ce_weights=[ce_weights],
        ref_kl_weights=None,
        rl_loss_fn=setup_rl_loss_fn(DefaultLossConfig()),
        rl_scale=scales[0],
        ce_scale=scales[1],
        ref_kl_scale=scales[2],
    )
    loss.backward()
    actual_loss = loss.detach().clone()
    dist.all_reduce(actual_loss)
    dist.all_reduce(theta.grad)
    drift = math.log(0.4 / 0.3)
    expected_loss = (-4 * (0.4 / 0.3) + 0.001 * 3.75 * drift**2) / 3 - 2 * math.log(0.4)
    expected_gradient = ((-4 * (0.4 / 0.3) + 0.002 * 3.75 * drift) / 3 - 2) * 0.6
    torch.testing.assert_close(actual_loss, actual_loss.new_tensor(expected_loss), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(theta.grad, theta.grad.new_tensor(expected_gradient), rtol=1e-10, atol=1e-10)
    tensors = Tensors()
    if world.is_master:
        tensors["probe/large"].append(torch.tensor([0.0, 1e30], dtype=torch.float32))
    if world.rank == world.world_size - 1:
        tensors["probe/other"].append(torch.tensor([0.25, 0.75], dtype=torch.float64))
    stats = tensors.compute_stats()
    assert all(math.isfinite(value) for value in stats.values()), stats
    assert math.isclose(stats["probe/large/mean"], 5e29, rel_tol=1e-6)
    assert stats["probe/other/mean"] == 0.5
    summaries = TensorSummaries(device=device)
    summaries.add("empty", torch.empty(0, device=device))
    if world.is_master:
        summaries.add("tail", torch.zeros(1, device=device))
        summaries.add("large", torch.tensor([0.0, 1e30], dtype=torch.float64, device=device))
        values = torch.tensor([float("nan")], device=device)
        summaries.add("nonfinite/raw", values)
        summaries.add("nonfinite/fraction", ~values.isfinite())
    if world.rank == world.world_size - 1:
        summaries.add("tail", torch.ones(9, device=device))
        summaries.add("accepted", torch.tensor([0.0, 1.0], device=device))
        values = torch.tensor([-float("inf"), float("inf"), 0.0], device=device)
        summaries.add("nonfinite/raw", values)
        summaries.add("nonfinite/fraction", ~values.isfinite())
    else:
        summaries.add("accepted", torch.empty(0, device=device))
    assert all(row.numel() == 4 for row in summaries.summaries.values())
    summary_stats = summaries.compute_stats()
    assert summaries.summaries == {}
    assert summary_stats["tail/count"] == 10
    assert summary_stats["tail/mean"] == 0.9  # A mean of rank means would incorrectly yield 0.5.
    assert summary_stats["accepted/count"] == 2
    assert summary_stats["accepted/mean"] == 0.5
    assert summary_stats["large/count"] == 2
    assert summary_stats["large/mean"] == 5e29
    assert summary_stats["large/max"] == 1e30
    assert summary_stats["empty/count"] == 0
    assert all(math.isnan(summary_stats[f"empty/{stat}"]) for stat in ("mean", "min", "max"))
    assert summary_stats["nonfinite/raw/count"] == 4
    assert math.isnan(summary_stats["nonfinite/raw/mean"])
    assert summary_stats["nonfinite/fraction/count"] == 4
    assert summary_stats["nonfinite/fraction/mean"] == 0.75
    assert all(
        math.isfinite(value) for key, value in summary_stats.items() if not key.startswith(("empty/", "nonfinite/raw/"))
    )
    assert not any(key.endswith(("/median", "/std")) for key in summary_stats)
    if world.is_master:
        print(
            json.dumps(
                {
                    "world_size": world.world_size,
                    "scales": scales,
                    "loss": actual_loss.item(),
                    "expected_loss": expected_loss,
                    "gradient": theta.grad.item(),
                    "expected_gradient": expected_gradient,
                    "diagnostics": stats,
                    "streaming_diagnostics": summary_stats,
                },
                indent=2,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
