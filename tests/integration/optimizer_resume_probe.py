"""Real DCP + CPU-state-offload resume, including Muon scalar groups and Adam control."""

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

import dion
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save

from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.optim import CPUOffloadOptimizer


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(item) for item in value)
    return copy.deepcopy(value)


def assert_equal(left, right, path="root"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_equal(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_equal(a, b, f"{path}/{index}")
    else:
        assert left == right, (path, left, right)


def setup(case):
    torch.manual_seed(13)
    model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 8)).cuda()
    parameters = list(model.parameters())
    if case == "adam":
        base = torch.optim.AdamW(
            [
                {"params": parameters[:2], "lr": 0.01, "group_label": "first"},
                {"params": parameters[2:], "lr": 0.02, "group_label": "second"},
            ],
            weight_decay=0,
        )
        optimizers = [CPUOffloadOptimizer(base)]
    else:
        selected = parameters if case == "muon" else parameters[:2]
        groups = [
            {
                "params": [p for p in selected if p.ndim == 2],
                "algorithm": "muon",
                "group_label": "matrices",
            },
            {
                "params": [p for p in selected if p.ndim == 1],
                "algorithm": "adamw",
                "group_label": "vectors",
            },
        ]
        muon = dion.Muon(
            groups,
            lr=0.01,
            weight_decay=0,
            fsdp_mesh_dim=0,
            world_mesh=None,
            adjust_lr="rms_norm",
        )
        optimizers = [CPUOffloadOptimizer(muon)]
        if case == "mixed":
            optimizers.append(
                torch.optim.AdamW(
                    [
                        {
                            "params": [parameters[2]],
                            "lr": 0.02,
                            "group_label": "other-matrix",
                        },
                        {
                            "params": [parameters[3]],
                            "lr": 0.03,
                            "group_label": "other-vector",
                        },
                    ],
                    weight_decay=0,
                )
            )
    return model, optimizers


def assert_offloaded(optimizers):
    for optimizer in optimizers:
        if isinstance(optimizer, CPUOffloadOptimizer):
            assert optimizer._initialized
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        assert value.device.type == "cpu" and value.is_pinned()


def snapshot(model, optimizers):
    names = {id(p): name for name, p in model.named_parameters()}
    result = {"model": clone(model.state_dict()), "optimizers": []}
    for optimizer in optimizers:
        result["optimizers"].append(
            {
                "groups": [
                    {
                        **clone({k: v for k, v in group.items() if k != "params"}),
                        "params": [names[id(p)] for p in group["params"]],
                    }
                    for group in optimizer.param_groups
                ],
                "state": {names[id(p)]: clone(state) for p, state in optimizer.state.items()},
            }
        )
    return result


def step(model, optimizers, seed):
    torch.manual_seed(seed)
    x = torch.randn(8, 16, device="cuda")
    loss = model(x).square().mean()
    loss.backward()
    gradients = {name: clone(p.grad) for name, p in model.named_parameters()}
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
    assert sum(gradient.norm() for gradient in gradients.values()) > 0
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad()
    assert_offloaded(optimizers)
    return {
        "loss": clone(loss),
        "gradients": gradients,
        "state": snapshot(model, optimizers),
    }


class ObservedLoad(AppState):
    def state_dict(self):
        state = super().state_dict()
        self.parameter_ids = [
            [id(p) for group in optimizer.param_groups for p in group["params"]] for optimizer in self.optimizers
        ]
        self.cpu_state_pointers = {
            (index, id(p), key): value.data_ptr()
            for index, optimizer in enumerate(self.optimizers)
            if isinstance(optimizer, CPUOffloadOptimizer)
            for p, values in optimizer.state.items()
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        }
        assert self.cpu_state_pointers
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        assert self.parameter_ids == [
            [id(p) for group in optimizer.param_groups for p in group["params"]] for optimizer in self.optimizers
        ]
        actual = {
            (index, id(p), key): value.data_ptr()
            for index, optimizer in enumerate(self.optimizers)
            if isinstance(optimizer, CPUOffloadOptimizer)
            for p, values in optimizer.state.items()
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        }
        assert actual == self.cpu_state_pointers, "DCP CPU state buffers must remain loaded in place"
        assert_offloaded(self.optimizers)


def run_case(case, output, expect):
    model, optimizers = setup(case)
    for seed in (31, 32, 33):
        step(model, optimizers, seed)
    if case != "adam":
        for optimizer_index, optimizer in enumerate(optimizers):
            for group_index, group in enumerate(optimizer.param_groups):
                group["lr"] *= 0.7 + 0.05 * (optimizer_index + group_index)
                group["group_label"] = f"saved/{optimizer_index}/{group_index}"
    saved = snapshot(model, optimizers)
    checkpoint = output / case / "checkpoint"
    dcp_save({"app": AppState(model, optimizers, None, None)}, checkpoint_id=checkpoint)
    uninterrupted = [step(model, optimizers, seed) for seed in (34, 35)]
    resumed_model, resumed_optimizers = setup(case)
    load_observer = ObservedLoad(resumed_model, resumed_optimizers, None, None)
    dcp_load({"app": load_observer}, checkpoint_id=checkpoint)
    loaded = snapshot(resumed_model, resumed_optimizers)
    assert_equal(saved["model"], loaded["model"], "loaded model")
    for index, expected in enumerate(saved["optimizers"]):
        assert_equal(
            expected["state"],
            loaded["optimizers"][index]["state"],
            f"loaded tensor-state/{index}",
        )
    continued = [step(resumed_model, resumed_optimizers, seed) for seed in (34, 35)]
    expected_steps = [group.get("step") for group in saved["optimizers"][0]["groups"]]
    loaded_steps = [group.get("step") for group in loaded["optimizers"][0]["groups"]]
    if expect == "frozen-bug" and case != "adam":
        assert expected_steps == [3, 3] and loaded_steps == [1, 1], (
            expected_steps,
            loaded_steps,
        )
        assert any(
            not torch.equal(
                uninterrupted[-1]["state"]["model"][name],
                continued[-1]["state"]["model"][name],
            )
            for name in saved["model"]
        ), "frozen reset must produce an actual update divergence"
        outcome = "reproduced scalar-step reset and continuation divergence"
    else:
        assert_equal(saved, loaded, "loaded complete state")
        assert_equal(uninterrupted, continued, "continued gradients/updates/state")
        outcome = "exact full-state resume and two continued updates"
    torch.save(
        {
            "saved": saved,
            "loaded": loaded,
            "uninterrupted": uninterrupted,
            "continued": continued,
        },
        output / case / "states.pt",
    )
    return {
        "outcome": outcome,
        "expected_muon_group_steps": expected_steps,
        "loaded_group_steps": loaded_steps,
        "parameter_identities_preserved": True,
        "cpu_state_buffers_loaded_in_place": True,
        "optimizer_count": len(optimizers),
    }


def rejection_controls():
    results = {}
    for corruption in ("count", "order", "membership"):
        model, optimizers = setup("mixed")
        app = AppState(model, optimizers, None, None)
        state = app.state_dict()
        before = snapshot(model, optimizers)
        assert not optimizers[0]._initialized
        groups = state["optimizers"]["param_groups"]
        if corruption == "count":
            groups.pop()
        elif corruption == "order":
            groups[0], groups[1] = groups[1], groups[0]
        else:
            groups[0]["params"] = ["not_a_model_parameter"]
        try:
            app.load_state_dict(state)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{corruption} mismatch accepted")
        assert not optimizers[0]._initialized
        assert_equal(before, snapshot(model, optimizers), f"rejection/{corruption}")
        results[corruption] = "rejected before wrapper initialization or metadata mutation"
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("frozen-bug", "fixed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 1
    sources = {
        "checkpoint": Path(inspect.getfile(AppState)).resolve(),
        "offload_optimizer": Path(inspect.getfile(CPUOffloadOptimizer)).resolve(),
    }
    assert all(path.is_relative_to(args.source_root.resolve()) for path in sources.values())
    args.output.mkdir(parents=True)
    results = {case: run_case(case, args.output, args.expect) for case in ("muon", "adam", "mixed")}
    receipt = {
        "rejection_controls": rejection_controls() if args.expect == "fixed" else None,
        "expectation": args.expect,
        "cases": results,
        "source_files": {
            key: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in sources.items()
        },
        "dion_source": str(Path(inspect.getfile(dion.Muon)).resolve()),
        "dion_sha256": hashlib.sha256(Path(inspect.getfile(dion.Muon)).read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Real single-rank DCP with FP32 MLP and actual CPUOffloadOptimizer/Dion Muon; separate Adam control and mixed wrapped/unwrapped optimizers. No full-model/FSDP repartition or throughput claim.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
