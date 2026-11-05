"""Run with torchrun at one and two ranks, then compare saved numerical states.

This exercises the real value runtime, FSDP, optimizer and checkpoint manager;
it deliberately does not launch replay transport or an evaluator service.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import LlamaConfig, LlamaForCausalLM

from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import setup_logger
from prime_rl.value.trainer import ValueTrainerRuntime
from prime_rl.value.types import ValueTrainingBatch, ValueTrainingSample


def _batch():
    # Three packed bins: two ranks require a dummy microbatch. Masks deliberately
    # differ in density so averaging rank means is not the global token mean.
    samples = [
        ValueTrainingSample(
            [1, 2, 3, 4, 5, 6, 7], [False, True, False, True, True, False, True], [0, -0.5, 0, 0.25, 0.75, 0, -0.25]
        ),
        ValueTrainingSample([8, 9, 10, 11, 12, 13], [False, False, False, True, False, True], [0, 0, 0, 0.5, 0, -0.75]),
        ValueTrainingSample([14, 15, 16, 17, 18], [False, True, True, True, True], [0, 0.25, -0.25, 0.75, -0.5]),
    ]
    return ValueTrainingBatch(
        samples=samples,
        num_rollouts=3,
        rollout_id_min=0,
        rollout_id_max=2,
        policy_version_min=0,
        policy_version_max=0,
        value_version_min=0,
        value_version_max=0,
        replay_attempt_min=1,
        replay_attempt_max=1,
        replay_attempt_mean=1.0,
    )


def _cpu(tensor):
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float().cpu().clone()


def _snapshot(runtime):
    model = {name: _cpu(value) for name, value in runtime.model.state_dict().items()}
    moments = {}
    for name, parameter in runtime.model.named_parameters():
        for key, value in runtime.optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor):
                moments[f"{name}/{key}"] = _cpu(value)
    return {
        "model": model,
        "moments": moments,
        "version": runtime.version,
        "total_tokens": runtime.progress.total_tokens,
        "total_samples": runtime.progress.total_samples,
        "lr": runtime.optimizer.param_groups[0]["lr"],
        "scheduler": runtime.scheduler.state_dict(),
    }


def _compare(left, right, *, rtol, atol):
    for key in ("version", "total_tokens", "total_samples", "lr", "scheduler"):
        assert left[key] == right[key], (key, left[key], right[key])
    measurements = {}
    for group in ("model", "moments"):
        assert left[group].keys() == right[group].keys()
        max_relative = 0.0
        max_absolute = 0.0
        for name, expected in left[group].items():
            actual = right[group][name]
            delta = (actual - expected).norm().item()
            reference_norm = expected.norm().item()
            assert torch.isfinite(actual).all(), name
            assert delta <= rtol * reference_norm + atol, (group, name, delta, reference_norm)
            max_relative = max(max_relative, delta / max(reference_norm, 1e-12))
            max_absolute = max(max_absolute, (actual - expected).abs().max().item())
        measurements[group] = {"max_parameter_relative_l2": max_relative, "max_element_absolute": max_absolute}
    return measurements


def _assert_update(initial, updated):
    assert updated["moments"], "optimizer state was not materialized"
    first_moments = [value for name, value in updated["moments"].items() if name.endswith("/exp_avg")]
    second_moments = [value for name, value in updated["moments"].items() if name.endswith("/exp_avg_sq")]
    assert first_moments and len(first_moments) == len(second_moments)
    assert sum(value.norm().item() for value in first_moments) > 0
    assert sum(value.norm().item() for value in second_moments) > 0
    assert any(not torch.equal(value, updated["model"][name]) for name, value in initial["model"].items())


def _compare_updates(initial_left, initial_right, left, right):
    # Parameter norms can dwarf optimizer steps. Compare the updates themselves
    # as well as the full states and Adam moments.
    maximum = 0.0
    for name in left["model"]:
        expected = left["model"][name] - initial_left["model"][name]
        actual = right["model"][name] - initial_right["model"][name]
        delta = (actual - expected).norm().item()
        norm = expected.norm().item()
        assert delta <= 0.025 * norm + 1e-6, ("update", name, delta, norm)
        maximum = max(maximum, delta / max(norm, 1e-12))
    return maximum


def run(output, resume):
    world = get_world()
    setup_logger("info")
    setup_torch_distributed()
    torch.set_float32_matmul_precision("high")
    if resume:
        resume_run(output, world)
        dist.barrier()
        dist.destroy_process_group()
        return
    output.mkdir(parents=True, exist_ok=True)
    seed_path = output / "seed-model"
    if world.is_master:
        torch.manual_seed(734)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=192,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        LlamaForCausalLM(config).save_pretrained(seed_path)
    dist.barrier()
    config = ValueFunctionConfig.model_validate(
        {
            "model": {
                "name": str(seed_path),
                "seq_len": 8,
                "attn": "flash_attention_2",
                "impl": "custom",
                "ep": 1,
                "compile": None,
                "ac": None,
                "ac_offloading": None,
                "optim_cpu_offload": False,
            },
            "loss": {"type": "mse"},
            "optim": {"type": "adamw", "lr": 1e-3, "weight_decay": 0, "max_norm": None},
            "scheduler": {"type": "cosine", "warmup_steps": 0, "min_lr": 5e-4},
            "output_dir": str(output / "runtime"),
            "batch_size": 3,
            "max_steps": 2,
            "ckpt": {"interval": 1, "keep_last": 3},
        }
    )
    assert config.model.optimization_dtype == config.model.reduce_dtype == "float32"
    if world.is_master:
        (output / "config.json").write_text(config.model_dump_json(indent=2))
    dims = get_parallel_dims(config.model)
    runtime = ValueTrainerRuntime(config, dims)
    # Load-independent head initialization eliminates rank-local RNG layout as
    # a confounder; the real pretrained backbone and runtime setup stay intact.
    with torch.no_grad():
        runtime.model.value_head.weight.fill_(0.01)
        runtime.model.value_head.bias.fill_(0.02)
    initial = _snapshot(runtime)
    batch = _batch()
    assert runtime.prepare_batch(batch)
    assert runtime._active.scale == 10
    layout = [micro.sample_indices for micro in runtime._active.micro_batches]
    layouts = [None] * world.world_size
    dist.all_gather_object(layouts, layout)
    if world.world_size == 2:
        assert any([-1] in rank for rank in layouts), layouts
    assert runtime.step() == 1
    step1 = _snapshot(runtime)
    _assert_update(initial, step1)
    assert runtime.prepare_batch(batch)
    assert runtime.step() == 2
    uninterrupted = _snapshot(runtime)
    assert uninterrupted["total_tokens"] == 20
    assert uninterrupted["total_samples"] == 6
    runtime.finish()
    if world.is_master:
        for name, state in (("initial", initial), ("step1", step1), ("step2", uninterrupted)):
            torch.save(state, output / f"{name}.pt")
        (output / "result.json").write_text(
            json.dumps(
                {
                    "world_size": world.world_size,
                    "layouts": layouts,
                    "total_tokens": 20,
                    "total_samples": 6,
                },
                indent=2,
            )
        )
    dist.barrier()
    dist.destroy_process_group()


def resume_run(output, world):
    config = ValueFunctionConfig.model_validate_json((output / "config.json").read_text())
    original = json.loads((output / "result.json").read_text())
    assert original["world_size"] == world.world_size
    config.ckpt.resume_step = 1
    step1 = torch.load(output / "step1.pt", weights_only=False)
    step2 = torch.load(output / "step2.pt", weights_only=False)
    runtime = ValueTrainerRuntime(config, get_parallel_dims(config.model))
    loaded = _snapshot(runtime)
    load_error = _compare(step1, loaded, rtol=0, atol=0)
    assert runtime.prepare_batch(_batch())
    assert runtime._active.scale == 10
    assert runtime.step() == 2
    restored = _snapshot(runtime)
    resume_error = _compare(step2, restored, rtol=1e-5, atol=1e-7)
    runtime.finish()
    if world.is_master:
        (output / "resume-result.json").write_text(
            json.dumps(
                {
                    "world_size": world.world_size,
                    "checkpoint_restore": load_error,
                    "resume_step2": resume_error,
                    "total_tokens": restored["total_tokens"],
                    "total_samples": restored["total_samples"],
                },
                indent=2,
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.compare:
        if args.resume:
            parser.error("--resume and --compare are separate operations")
        originals = [json.loads((path / "result.json").read_text()) for path in args.compare]
        assert {result["world_size"] for result in originals} == {1, 2}, originals
        results = {"resume": [json.loads((path / "resume-result.json").read_text()) for path in args.compare]}
        for original, resumed in zip(originals, results["resume"], strict=True):
            assert original["world_size"] == resumed["world_size"]
        initial_left, initial_right = [torch.load(path / "initial.pt", weights_only=False) for path in args.compare]
        for name in ("initial", "step1", "step2"):
            left, right = [torch.load(path / f"{name}.pt", weights_only=False) for path in args.compare]
            results[name] = _compare(
                left, right, rtol=0 if name == "initial" else 0.025, atol=0 if name == "initial" else 1e-6
            )
            if name != "initial":
                _assert_update(initial_left, left)
                _assert_update(initial_right, right)
                results[name]["max_update_relative_l2"] = _compare_updates(initial_left, initial_right, left, right)
        print(json.dumps(results, indent=2))
    else:
        if args.output is None:
            parser.error("--output is required when not comparing")
        run(args.output, args.resume)


if __name__ == "__main__":
    main()
