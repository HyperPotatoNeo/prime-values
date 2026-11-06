"""Real compiled EP/FSDP expert saves, with independent alias/offload observations."""

import argparse
import hashlib
import inspect
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torchtitan.distributed.expert_parallel import ExpertParallel

import prime_rl.utils.act_offloading as offloading
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.parallel_dims import ParallelDims


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def storages(model):
    with torch.no_grad():
        return {local(p).untyped_storage()._cdata for p in model.parameters()}


def observer(model, mode, records):
    if mode == "off":
        return nullcontext()
    hook = offloading.OffloadActivations(model) if mode == "candidate" else offloading.OffloadActivations()
    assert hook.use_streams is False and hook.use_pin_memory is True
    original_pack = hook.pack_hook
    initial_storages = storages(model)

    def observed_pack(tensor):
        # Never retain a tensor/storage reference in the observer: doing so
        # would alter the FSDP allocation lifecycle under test.
        identity = local(tensor).untyped_storage()._cdata
        is_live_parameter = identity in storages(model)
        eligible = tensor.device.type == "cuda" and tensor.numel() * tensor.element_size() >= 1024
        token = original_pack(tensor)
        saved, copied = hook.tracker[token]
        records.append(
            {
                "parameter_alias": is_live_parameter,
                "expert_alias": identity in storages(model.experts),
                "plain_tensor": not isinstance(tensor, nn.Parameter),
                "eligible": eligible,
                "changed_storage": is_live_parameter and identity not in initial_storages,
                "copied": copied,
                "saved_device": saved.device.type,
                "bytes": tensor.numel() * tensor.element_size(),
            }
        )
        return token

    hook.pack_hook = observed_pack
    return hook


def replacement_control():
    model = nn.Linear(32, 32, bias=False, device="cuda")
    hook = offloading.OffloadActivations(model)
    old_parameter = model.weight
    model.weight = nn.Parameter(old_parameter.detach().clone())
    assert old_parameter.untyped_storage()._cdata != model.weight.untyped_storage()._cdata
    live_storage = model.weight.untyped_storage()._cdata
    activation = model.weight.detach().clone().requires_grad_()
    original_pack = hook.pack_hook
    observations = []

    def pack(tensor):
        identity = tensor.untyped_storage()._cdata
        token = original_pack(tensor)
        observations.append((identity, hook.tracker[token][1]))
        return token

    hook.pack_hook = pack
    x = torch.randn(64, 32, device="cuda", requires_grad=True)
    with hook:
        loss = model(x).square().mean() + (activation @ x.T).square().mean()
    assert any(identity == live_storage and not copied for identity, copied in observations)
    assert not any(identity == live_storage and copied for identity, copied in observations)
    assert any(identity == activation.untyped_storage()._cdata and copied for identity, copied in observations)
    loss.backward()
    assert torch.isfinite(model.weight.grad).all() and model.weight.grad.norm() > 0
    return {"replaced_parameter_retained": True, "equal_valued_activation_clone_offloaded": True}


class ExpertBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = GroupedExperts(dim=64, hidden_dim=128, num_experts=4, use_grouped_mm=True)
        self.projection = nn.Linear(64, 64, bias=False)

    def forward(self, x, counts):
        return self.projection(self.experts(x, counts))


def full(tensor):
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().cpu().clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("off", "frozen", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    sources = {
        "offloading": Path(offloading.__file__).resolve(),
        "experts": Path(inspect.getfile(GroupedExperts)).resolve(),
    }
    assert all(path.is_relative_to(args.source_root.resolve()) for path in sources.values())
    control = replacement_control() if args.mode == "candidate" else None
    torch.manual_seed(2381)
    torch.set_float32_matmul_precision("high")
    block = ExpertBlock().cuda()
    block.experts.init_weights(0.02)
    assert all(torch.isfinite(p).all() for p in block.parameters())
    assert not torch.equal(block.experts.w1[0], block.experts.w1[1])
    dims = ParallelDims(dp_replicate=1, dp_shard=-1, cp=1, pp=1, ep=2, world_size=2)
    parallelize_module(block.experts, device_mesh=dims.get_mesh("ep"), parallelize_plan=ExpertParallel())
    block.compile(fullgraph=False)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(block.experts, mesh=dims.world_mesh["dp_shard_mod_ep"], mp_policy=policy, reshard_after_forward=True)
    fully_shard(block, mesh=dims.get_mesh("hsdp"), mp_policy=policy, reshard_after_forward=True)
    assert isinstance(block.experts.w1, DTensor)
    assert all(p.dtype == torch.float32 for p in block.parameters())
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3, weight_decay=0)
    tensors = {f"initial/{name}": full(p) for name, p in block.named_parameters()}
    records = []
    captured = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    for step in range(2):
        torch.manual_seed(100 + rank + step)
        x = torch.randn(32, 64, device="cuda", requires_grad=True)
        counts = torch.full((4,), 8, dtype=torch.int64, device="cuda")
        with observer(block, args.mode, records):
            output = block(x, counts)
            loss = output.float().square().mean()
        loss.backward()
        tensors[f"step{step}/output"] = full(output)
        tensors[f"step{step}/input_grad"] = full(x.grad)
        for name, p in block.named_parameters():
            assert p.grad is not None
            tensors[f"step{step}/gradient/{name}"] = full(p.grad)
        optimizer.step()
        for name, p in block.named_parameters():
            tensors[f"step{step}/parameter/{name}"] = full(p)
            for key, value in optimizer.state[p].items():
                if isinstance(value, torch.Tensor):
                    tensors[f"step{step}/optimizer/{name}/{key}"] = full(value)
        optimizer.zero_grad()
    graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) - captured
    assert graphs > 0
    assert all(torch.isfinite(value).all() for value in tensors.values())
    assert any(not torch.equal(tensors[f"initial/{name}"], full(p)) for name, p in block.named_parameters())
    aliases = [row for row in records if row["expert_alias"] and row["plain_tensor"] and row["eligible"]]
    if args.mode != "off":
        assert aliases, "real compiled expert graph must save plain live parameter aliases"
        assert any(row["changed_storage"] for row in aliases), "must observe live FSDP storage replacement"
        assert any(row["copied"] and not row["parameter_alias"] for row in records), "real activations must offload"
        if args.mode == "candidate":
            assert all(not row["copied"] and row["saved_device"] == "cuda" for row in records if row["parameter_alias"])
        else:
            assert any(row["copied"] for row in aliases), "frozen negative control must expose weight copies"
    if args.reference:
        reference_receipt = json.loads((args.reference / f"rank{rank}.json").read_text())
        assert reference_receipt["harness_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        expected = torch.load(args.reference / f"rank{rank}.pt", weights_only=True)
        assert tensors.keys() == expected.keys()
        for key, value in tensors.items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0, msg=key)
    if rank == 0:
        args.output.mkdir(parents=True)
    dist.barrier()
    torch.save(tensors, args.output / f"rank{rank}.pt")
    receipt = {
        "mode": args.mode,
        "rank": rank,
        "compiled_graphs": graphs,
        "observations": records,
        "replacement_control": control,
        "reference": str(args.reference),
        "source_files": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in sources.items()
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Actual GroupedExperts, torch EP2 and nested FSDP with compiled block; FP32 masters/reduction/Adam, BF16 forward. Not a full model or throughput benchmark.",
    }
    (args.output / f"rank{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
