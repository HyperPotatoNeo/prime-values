"""Two-rank real dense Qwen CP oracle; launch separately for ring and Ulysses.

Run the position mode against frozen source first, then explicit mode against
the candidate. Each process computes an independent unsharded reference before
patching attention for CP. No claim about hybrid-model packed CP is made.
"""

import argparse
import copy
import functools
import inspect
import os
from pathlib import Path

import torch
import torch.distributed as dist
from model_oracle import actor_loss, compare_gradients_and_step, make_model, metadata_kwargs, save_result, tensor_inputs

from prime_rl.trainer.model import forward
from prime_rl.trainer.models.layers.attn import substitute_ring_attn
from prime_rl.trainer.models.layers.ulysses_attn import substitute_ulysses_attn
from prime_rl.utils.cp import gather_for_cp, setup_cp_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["position", "explicit"], required=True)
    parser.add_argument("--style", choices=["ring", "ulysses"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    # Rank 1 starts at offset 8, the third token of the second document.
    # A separate final padding segment is preserved, with no supervised token.
    lengths = [6, 5, 3, 2]
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0]
    masks = [False, False, True, True, False, True, False, False, True, True, True, False, True, True, False, False]
    ids, positions = tensor_inputs(tokens, lengths)
    model = make_model()
    reference = copy.deepcopy(model)
    expected = forward(reference, ids, positions)["logits"]
    expected_loss = actor_loss(expected, tokens, masks, lengths, sum(masks))
    expected_loss.backward()

    if args.style == "ring":
        import ring_flash_attn

        kernel = ring_flash_attn.llama3_flash_attn_varlen_func
        assert "deterministic" in inspect.signature(kernel).parameters
        ring_flash_attn.llama3_flash_attn_varlen_func = functools.partial(kernel, deterministic=True)
        substitute_ring_attn(dist.group.WORLD, heads_k_stride=1)
    else:
        import flash_attn

        kernel = flash_attn.flash_attn_varlen_func
        assert "deterministic" in inspect.signature(kernel).parameters
        flash_attn.flash_attn_varlen_func = functools.partial(kernel, deterministic=True)
        substitute_ulysses_attn(dist.group.WORLD)
    global_kwargs = metadata_kwargs(lengths, args.mode)
    local_ids, local_positions = setup_cp_params(
        ids, positions, rank, 2, dist.group.WORLD, cp_style=args.style, **global_kwargs
    )
    local_kwargs = {}
    if global_kwargs:
        local = global_kwargs["packing_metadata"].for_cp(rank, 2)
        assert local.lengths == ((6, 2) if rank == 0 else (3, 3, 2))
        local_kwargs = {"packing_metadata": local}
    assert local_positions[0, 0].item() == (0 if rank == 0 else 2)
    local_logits = forward(model, local_ids, local_positions, **local_kwargs)["logits"]
    logits = gather_for_cp(local_logits, dist.group.WORLD)
    torch.testing.assert_close(logits, expected, rtol=2e-3, atol=2e-3)
    full_loss = actor_loss(logits, tokens, masks, lengths, sum(masks))
    torch.testing.assert_close(full_loss, expected_loss, rtol=2e-3, atol=2e-4)
    # All ranks evaluate the global objective; differentiable gather sums its
    # backward contributions. Divide by ranks, then sum replicated parameter
    # gradients to obtain the same objective as the single-model reference.
    (full_loss / 2).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    tensors = {"cp/logits": logits.detach().cpu(), "cp/loss": full_loss.detach().cpu()}
    metrics = compare_gradients_and_step(model, reference, tensors, "cp")
    metrics.update({"style": args.style, "world_size": 2, "lengths": lengths})
    if rank == 0:
        save_result(args.output, args.mode, tensors, metrics, args.reference)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
