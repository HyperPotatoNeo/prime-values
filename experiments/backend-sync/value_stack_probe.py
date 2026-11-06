"""Run with torchrun at one and two ranks, then compare saved numerical states.

This exercises the real value runtime, FSDP, optimizer and checkpoint manager;
it deliberately does not launch replay transport or an evaluator service.
"""

import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

import msgspec
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import LlamaConfig, LlamaForCausalLM

import prime_rl.trainer.model as model_module
import prime_rl.trainer.models.layers.attn as attention_module
import prime_rl.trainer.models.llama.modeling_llama as llama_module
import prime_rl.utils.act_offloading as offloading_module
import prime_rl.utils.sequence as sequence_module
import prime_rl.value.batch as batch_module
import prime_rl.value.trainer as value_trainer_module
from prime_rl.configs.trainer import ActivationCheckpointConfig, ActivationOffloadingConfig, CompileConfig
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
    result = {
        "model": model,
        "moments": moments,
        "version": runtime.version,
        "total_tokens": runtime.progress.total_tokens,
        "total_samples": runtime.progress.total_samples,
        "lr": runtime.optimizer.param_groups[0]["lr"],
        "scheduler": runtime.scheduler.state_dict(),
    }
    if hasattr(runtime, "_probe_gradients"):
        result["gradients"] = runtime._probe_gradients
    return result


def _compare(left, right, *, rtol, atol):
    for key in ("version", "total_tokens", "total_samples", "lr", "scheduler"):
        assert left[key] == right[key], (key, left[key], right[key])
    measurements = {}
    assert ("gradients" in left) == ("gradients" in right)
    for group in ("model", "moments") + (("gradients",) if "gradients" in left else ()):
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


def _tensor_difference(expected, actual, *, rtol, atol, max_absolute=None):
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    delta = actual.double() - expected.double()
    error = delta.norm().item()
    reference = expected.double().norm().item()
    largest = delta.abs().max().item() if delta.numel() else 0.0
    bound = rtol * reference + atol
    return {
        "reference_l2": reference,
        "difference_l2": error,
        "relative_l2": error / reference if reference else None,
        "max_element_absolute": largest,
        "l2_bound": bound,
        "max_element_bound": max_absolute,
        "passed": error <= bound and (max_absolute is None or largest <= max_absolute),
    }


def compare_stacks(paths):
    """Apply the predeclared backend criteria to full saved numerical records."""
    records = [json.loads((path / "result.json").read_text()) for path in paths]
    configs = [json.loads((path / "config.json").read_text()) for path in paths]
    for path, record, config in zip(paths, records, configs, strict=True):
        assert record["acceptance_status"] == "passed", "incomplete diagnostic evidence cannot pass acceptance"
        assert record["world_size"] == 2 and record["compiled"]
        assert record["total_tokens"] == 20 and record["total_samples"] == 6
        assert len(record["rank_compile_receipts"]) == 2
        for rank in record["rank_compile_receipts"]:
            assert rank["compiled_layers"] == 2 and rank["captured_graphs"] > 0
            assert rank["moment_dtypes"] == ["torch.float32"]
        assert config["model"]["optimization_dtype"] == config["model"]["reduce_dtype"] == "float32"
        files = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path / "seed-model").iterdir() if p.is_file()
        }
        assert files == record["fixture_hashes"]
        assert "probe-batch.msgpack" in files and any(name.endswith(".safetensors") for name in files)
        # These are the only path differences permitted between paired runs.
        assert Path(config.pop("output_dir")).resolve() == (path / "runtime").resolve()
        assert Path(config["model"].pop("name")).resolve() == (path / "seed-model").resolve()
    assert configs[0] == configs[1], "resolved value configurations differ"
    for key in ("fixture_hashes", "layouts", "compile_config", "checkpoint_activations", "activation_offload"):
        assert records[0][key] == records[1][key], key
    assert any([-1] in rank for rank in records[0]["layouts"]), "dummy rank must be exercised"
    assert records[0]["sources"]["harness_sha256"] == records[1]["sources"]["harness_sha256"]
    assert records[0]["sources"]["harness_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    initial = [torch.load(path / "initial.pt", weights_only=False) for path in paths]
    _compare(*initial, rtol=0, atol=0)
    measurements = {}
    for step in ("step1", "step2"):
        states = [torch.load(path / f"{step}.pt", weights_only=False) for path in paths]
        for start, state in zip(initial, states, strict=True):
            _assert_update(start, state)
        for key in ("version", "total_tokens", "total_samples", "lr", "scheduler"):
            assert states[0][key] == states[1][key], (step, key)
        for group in ("gradients", "moments", "model"):
            left, right = states[0][group], states[1][group]
            assert left and left.keys() == right.keys(), (step, group)
            for name, expected in left.items():
                actual = right[name]
                if group == "model":
                    expected = expected - initial[0]["model"][name]
                    actual = actual - initial[1]["model"][name]
                exact = group == "moments" and name.endswith("/step")
                measurements[f"{step}/{group}/{name}"] = _tensor_difference(
                    expected, actual, rtol=0 if exact else 0.025, atol=0 if exact else 1e-6
                )
    for rank in range(2):
        outputs = [torch.load(path / f"outputs-rank{rank}.pt", weights_only=False) for path in paths]
        assert outputs[0] and len(outputs[0]) == len(outputs[1])
        for index, (left, right) in enumerate(zip(*outputs, strict=True)):
            assert left.keys() == right.keys()
            assert left["scale"] == right["scale"] == 10
            for name in ("mask", "targets"):
                assert left[name].dtype == right[name].dtype and torch.equal(left[name], right[name]), (
                    rank,
                    index,
                    name,
                )
            for name in ("logits", "loss", "logit_gradient"):
                gradient = name == "logit_gradient"
                measurements[f"rank{rank}/microbatch{index}/{name}"] = _tensor_difference(
                    left[name],
                    right[name],
                    rtol=0.025 if gradient else 0.01,
                    atol=1e-6 if gradient else 1e-5,
                    max_absolute=None if gradient else 0.005,
                )
    failures = [name for name, result in measurements.items() if not result["passed"]]
    report = {"passed": not failures, "failures": failures, "measurements": measurements}
    # Preserve every measured discrepancy even when a bound blocks acceptance.
    (paths[1] / "stack-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    assert not failures, f"Backend numerical criteria failed: {failures}"
    return {"passed": True, "compared_tensors": len(measurements), "report": str(paths[1] / "stack-comparison.json")}


def _source_receipt(expected_root):
    sources = {
        "runtime": Path(inspect.getfile(ValueTrainerRuntime)).resolve(),
        "offloading": Path(offloading_module.__file__).resolve(),
        "model": Path(model_module.__file__).resolve(),
        "llama_backbone": Path(llama_module.__file__).resolve(),
        "attention": Path(attention_module.__file__).resolve(),
        "sequence": Path(sequence_module.__file__).resolve(),
        "value_batch": Path(batch_module.__file__).resolve(),
        "config": Path(inspect.getfile(ValueFunctionConfig)).resolve(),
    }
    opaque_ops = sys.modules.get("prime_rl.trainer.models.layers.flash_attn_ops")
    if opaque_ops is not None:
        sources["flash_attn_ops"] = Path(opaque_ops.__file__).resolve()
    if expected_root is not None:
        expected_root = expected_root.resolve()
        assert all(path.is_relative_to(expected_root) for path in sources.values()), sources
    return {
        "expected_root": str(expected_root) if expected_root is not None else None,
        "files": {
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in sources.items()
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def run(
    output,
    resume,
    compiled=False,
    source_root=None,
    activation_offload=False,
    checkpoint_activations=False,
    seed_model=None,
):
    source_receipt = _source_receipt(source_root)
    if compiled and output.exists():
        raise FileExistsError(f"Compiled fresh run requires a new output directory: {output}")
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
        if seed_model is not None:
            seed_path.mkdir()
            for path in seed_model.iterdir():
                if path.is_file():
                    shutil.copy2(path, seed_path / path.name)
        else:
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
            (seed_path / "probe-batch.msgpack").write_bytes(msgspec.msgpack.encode(_batch()))
    dist.barrier()
    config = ValueFunctionConfig.model_validate(
        {
            "model": {
                "name": str(seed_path),
                "seq_len": 8,
                "attn": "flash_attention_2",
                "impl": "custom",
                "ep": 1,
                "compile": CompileConfig().model_dump() if compiled else None,
                "ac": ActivationCheckpointConfig().model_dump() if checkpoint_activations else None,
                "ac_offloading": ActivationOffloadingConfig().model_dump() if activation_offload else None,
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
    captured_before = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    offload_counts = []
    if activation_offload:
        original_offloader = offloading_module.OffloadActivations

        class ObservedOffloader(original_offloader):
            def __exit__(self, *args):
                offload_counts.append(sum(offloaded for _, offloaded in self.tracker.values()))
                assert self.use_streams is False
                return super().__exit__(*args)

        offloading_module.OffloadActivations = ObservedOffloader
    compile_setup = []
    original_apply_compile = model_module.apply_compile

    def observed_apply_compile(model, compile_config):
        result = original_apply_compile(model, compile_config)
        language_model = model_module.get_language_model(model)
        for layer in language_model.layers:
            compile_setup.append(
                {
                    "mro": [f"{cls.__module__}.{cls.__qualname__}" for cls in type(layer).__mro__],
                    "compiled_callable": repr(layer._compiled_call_impl),
                }
            )
        return result

    model_module.apply_compile = observed_apply_compile
    try:
        runtime = ValueTrainerRuntime(config, dims)
    finally:
        model_module.apply_compile = original_apply_compile
    compiled_call_counts = [0] * len(runtime.model.model.layers)
    layer_after_fsdp = []
    for index, layer in enumerate(runtime.model.model.layers):
        original_call = layer._compiled_call_impl
        layer_after_fsdp.append(
            {
                "mro": [f"{cls.__module__}.{cls.__qualname__}" for cls in type(layer).__mro__],
                "compiled_callable": repr(original_call),
            }
        )
        if original_call is not None:

            def counted_call(*args, _index=index, _original=original_call, **kwargs):
                compiled_call_counts[_index] += 1
                return _original(*args, **kwargs)

            layer._compiled_call_impl = counted_call
    original_step = runtime.optimizer.step

    def observed_step(*args, **kwargs):
        runtime._probe_gradients = {
            name: _cpu(parameter.grad)
            for name, parameter in runtime.model.named_parameters()
            if parameter.grad is not None
        }
        assert runtime._probe_gradients
        return original_step(*args, **kwargs)

    if compiled:
        runtime._probe_gradients = {}
        runtime.optimizer.step = observed_step
    compiled_layers = sum(layer._compiled_call_impl is not None for layer in runtime.model.model.layers)
    if compiled:
        assert compiled_layers == len(runtime.model.model.layers) == 2
        assert all(parameter.dtype == torch.float32 for parameter in runtime.model.parameters())
    # Load-independent head initialization eliminates rank-local RNG layout as
    # a confounder; the real pretrained backbone and runtime setup stay intact.
    with torch.no_grad():
        runtime.model.value_head.weight.fill_(0.01)
        runtime.model.value_head.bias.fill_(0.02)
    initial = _snapshot(runtime)
    fixture_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in seed_path.iterdir() if p.is_file()}
    batch = msgspec.msgpack.decode((seed_path / "probe-batch.msgpack").read_bytes(), type=ValueTrainingBatch)
    observed_outputs = []
    original_loss = value_trainer_module.compute_value_loss

    def observed_loss(logits, targets, mask, loss_config, scale):
        loss, metrics = original_loss(logits, targets, mask, loss_config, scale)
        derivative = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected = torch.zeros_like(logits)
        expected[..., 0][mask] = 2 * (logits[..., 0][mask] - targets[mask].float()) / scale
        torch.testing.assert_close(derivative, expected, atol=1e-7, rtol=1e-6)
        observed_outputs.append(
            {
                "logits": _cpu(logits),
                "mask": mask.detach().cpu().clone(),
                "targets": _cpu(targets),
                "loss": _cpu(loss),
                "logit_gradient": _cpu(derivative),
                "scale": scale,
            }
        )
        return loss, metrics

    value_trainer_module.compute_value_loss = observed_loss
    assert runtime.prepare_batch(batch)
    assert runtime._active.scale == 10
    layout = [micro.sample_indices for micro in runtime._active.micro_batches]
    layouts = [None] * world.world_size
    dist.all_gather_object(layouts, layout)
    if world.world_size == 2:
        assert any([-1] in rank for rank in layouts), layouts
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as execution_profile:
        assert runtime.step() == 1
        step1 = _snapshot(runtime)
        _assert_update(initial, step1)
        assert runtime.prepare_batch(batch)
        assert runtime.step() == 2
        uninterrupted = _snapshot(runtime)
        assert uninterrupted["total_tokens"] == 20
        assert uninterrupted["total_samples"] == 6
    compiled_graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) - captured_before
    moment_dtypes = sorted(
        {
            str(value.dtype)
            for state in runtime.optimizer.state.values()
            for key, value in state.items()
            if key in {"exp_avg", "exp_avg_sq"}
        }
    )
    compile_receipts = [None] * world.world_size
    dist.all_gather_object(
        compile_receipts,
        {
            "compiled_layers": compiled_layers,
            "captured_graphs": compiled_graphs,
            "moment_dtypes": moment_dtypes,
            "offloaded_counts": offload_counts,
            "dynamo_counters": {key: dict(value) for key, value in torch._dynamo.utils.counters.items()},
            "compiled_call_counts": compiled_call_counts,
            "compile_setup": compile_setup,
            "layers_after_fsdp": layer_after_fsdp,
            "compiler_environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith(("TORCH_COMPILE", "TORCHDYNAMO", "TORCHINDUCTOR"))
            },
            "dynamo_config": {
                "disable": torch._dynamo.config.disable,
                "suppress_errors": torch._dynamo.config.suppress_errors,
            },
            "compiler_profile_events": {
                event.key: event.count
                for event in execution_profile.key_averages()
                if any(word in event.key.lower() for word in ("compil", "dynamo", "inductor"))
            },
        },
    )
    # Persist real numerical evidence before diagnostic acceptance gates so a
    # failed compile observation never discards completed production updates.
    assert _source_receipt(source_root) == source_receipt
    torch.save(observed_outputs, output / f"outputs-rank{world.rank}.pt")
    if world.is_master:
        for name, state in (("initial", initial), ("step1", step1), ("step2", uninterrupted)):
            torch.save(state, output / f"{name}.pt")
        (output / "result.json").write_text(
            json.dumps(
                {
                    "acceptance_status": "pending_compile_and_lifecycle_checks",
                    "world_size": world.world_size,
                    "layouts": layouts,
                    "total_tokens": 20,
                    "total_samples": 6,
                    "compiled": compiled,
                    "activation_offload": activation_offload,
                    "checkpoint_activations": checkpoint_activations,
                    "rank_local_offloaded_counts": offload_counts,
                    "compile_config": config.model.compile.model_dump() if config.model.compile is not None else None,
                    "rank_compile_receipts": compile_receipts,
                    "sources": source_receipt,
                    "fixture_hashes": fixture_hashes,
                    "independent_mse_logit_gradient": "passed every actual microbatch including dummy",
                },
                indent=2,
            )
        )
    dist.barrier()
    if compiled:
        assert compiled_graphs > 0, "compiled blocks must actually capture and execute graphs"
        assert moment_dtypes == ["torch.float32"], moment_dtypes
    if activation_offload:
        assert offload_counts and sum(offload_counts) > 0, "the actual critic must offload real activations"
    runtime.finish()
    assert _source_receipt(source_root) == source_receipt
    if world.is_master:
        result = json.loads((output / "result.json").read_text())
        result["acceptance_status"] = "passed"
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
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
    parser.add_argument(
        "--seed-model",
        type=Path,
        help="Copy exact frozen HF initialweights and batch fixture; never regenerate candidate RNG weights",
    )
    parser.add_argument("--checkpoint-activations", action="store_true", help="Use normal full block checkpointing")
    parser.add_argument("--activation-offload", action="store_true", help="Enable normal activation-offload config")
    parser.add_argument("--compiled", action="store_true", help="Use the normal model-block compilation configuration")
    parser.add_argument(
        "--source-root", type=Path, help="Require runtime and configuration imports under this checkout"
    )
    parser.add_argument("--compare-compiled", type=Path, nargs=2, help="Compare frozen/candidate two-rank fresh states")
    parser.add_argument("--compare-stacks", type=Path, nargs=2, help="Apply predeclared cross-backend numerical bounds")
    args = parser.parse_args()
    if args.compare_stacks:
        if args.compare or args.compare_compiled or args.resume or args.compiled:
            parser.error("--compare-stacks is a separate operation")
        print(json.dumps(compare_stacks(args.compare_stacks), indent=2))
    elif args.compare_compiled:
        if args.compare or args.resume or args.compiled:
            parser.error("--compare-compiled is a separate operation")
        originals = [json.loads((path / "result.json").read_text()) for path in args.compare_compiled]
        assert all(result["world_size"] == 2 and result["compiled"] for result in originals), originals
        assert originals[0]["sources"]["harness_sha256"] == originals[1]["sources"]["harness_sha256"]
        resolved = [json.loads((path / "config.json").read_text()) for path in args.compare_compiled]
        assert resolved[0]["model"]["ac"] == resolved[1]["model"]["ac"], "checkpointing must match across controls"
        for config in resolved:
            assert config["model"]["optimization_dtype"] == config["model"]["reduce_dtype"] == "float32"
        assert originals[0]["compile_config"] == originals[1]["compile_config"]
        assert originals[0]["layouts"] == originals[1]["layouts"]
        results = {}
        initial_left, initial_right = [
            torch.load(path / "initial.pt", weights_only=False) for path in args.compare_compiled
        ]
        for name in ("initial", "step1", "step2"):
            left, right = [torch.load(path / f"{name}.pt", weights_only=False) for path in args.compare_compiled]
            results[name] = _compare(left, right, rtol=0, atol=0)
            if name != "initial":
                _assert_update(initial_left, left)
                _assert_update(initial_right, right)
                results[name]["max_update_relative_l2"] = _compare_updates(initial_left, initial_right, left, right)
        print(json.dumps(results, indent=2))
    elif args.compare:
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
        if args.compiled and (args.resume or args.source_root is None):
            parser.error("--compiled requires --source-root and a fresh run without --resume")
        if args.activation_offload and not (args.compiled and args.checkpoint_activations):
            parser.error("--activation-offload acceptance requires --compiled --checkpoint-activations")
        run(
            args.output,
            args.resume,
            compiled=args.compiled,
            source_root=args.source_root,
            activation_offload=args.activation_offload,
            checkpoint_activations=args.checkpoint_activations,
            seed_model=args.seed_model,
        )


if __name__ == "__main__":
    main()
