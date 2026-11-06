"""CPU actual response/worker/sink/packing oracle for observation-only version labels."""

import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import json
import multiprocessing as mp
import uuid
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import torch
import verifiers.v1 as vf
from aiohttp import web
from renderers.qwen3 import Qwen3Renderer
from rg_mix_v1.taskset import RGMixConfig, RGMixTaskset
from transformers import AutoTokenizer
from verifiers.v1.serve import EnvClient, EnvServer

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.transport.types import MicroBatch, TrainingSample


def load_module(name, path):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def worker(raw, queue):
    server = EnvServer(EnvConfig.model_validate(raw).env, address="tcp://127.0.0.1:0")
    queue.put(server.address)
    asyncio.run(server.run())


def old_fields(value):
    return {
        name: msgspec.to_builtins(getattr(value, name))
        for name in value.__struct_fields__
        if name != "token_policy_versions"
    }


def labels(value):
    payload = getattr(value, "token_policy_versions", None)
    if payload is None:
        return None
    assert payload.dtype == "int64" and payload.shape == [
        len(value.token_ids if isinstance(value, TrainingSample) else value.input_ids)
    ]
    return np.frombuffer(payload.data, dtype="<i8").tolist()


def tensors(micro):
    holder = SimpleNamespace(multi_run_manager=SimpleNamespace(max_runs=1))
    actual = DataLoader._micro_batch_to_tensor(holder, micro)
    versions = actual.pop("token_policy_versions", None)
    if versions is not None:
        assert (
            versions.dtype == torch.int64
            and versions.device.type == "cpu"
            and versions.shape == (1, len(micro.input_ids))
        )
    return {
        k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in actual.items()
    }, None if versions is None else versions.tolist()


async def finalize(group, config, tokenizer):
    requests = []
    prompts = []
    published = []

    class Evaluator:
        async def evaluate(self, rows):
            requests.append([list(row) for row in rows])
            return SimpleNamespace(values=[[0.0] + [0.01 * (x % 11) for x in row[1:]] for row in rows], version=7)

    class Renderer:
        def render_ids(self, messages, *, add_generation_prompt):
            prompts.append(messages[0]["content"])
            return Qwen3Renderer(tokenizer).render_ids(messages, add_generation_prompt=add_generation_prompt)

    envs = TrainEnvs(
        config.train.env, policy_pool=None, value_evaluator=Evaluator(), value_config=config.value_function
    )
    sink = TrainSink(
        config,
        tokenizer=tokenizer,
        renderer=Renderer(),
        train_envs=envs,
        mm_token_type_ids_mapping=None,
        batch_size=2,
        token_batch_size=None,
        pre_filters=[],
        post_filters=[],
        value_publisher=SimpleNamespace(publish=published.append),
    )
    copies = [r.model_copy(deep=True) for r in group]
    try:
        for r in copies:
            r.env_name = "rg"
            r.group_id = uuid.UUID(int=1)
            r.policy_version = 999
        assert await sink.add(copies[0]) is None
        if id(copies[0]) in sink.scoring_tasks:
            await sink.scoring_tasks[id(copies[0])]
        batch = await sink.add(copies[1])
        assert batch is not None
        numeric = {
            "requests": requests,
            "prompts": prompts,
            "published": [[msgspec.to_builtins(s) for s in x.samples] for x in published],
            "rollouts": [
                {
                    "values": r.value_predictions,
                    "advantages": r.value_advantages,
                    "targets": r.value_returns,
                    "reward": r.reward,
                }
                for r in copies
            ],
            "samples": [old_fields(s) for s in batch.samples],
        }
        numeric = json.loads(json.dumps(numeric))
        live = list(envs)[0].sampler.samples_from_live_policy
        saved_labels = [labels(s) for s in batch.samples]
        for r in copies:
            r.nodes.clear()
            r.samples.clear()
            r.rewards.clear()
        assert [[msgspec.to_builtins(s) for s in x.samples] for x in published] == numeric["published"]
        return batch.samples, numeric, saved_labels, live
    finally:
        await sink.stop()


def pack(samples):
    groups = prepare_batch(
        samples,
        seq_len=1536,
        num_train_workers=1,
        idxs=[0] * len(samples),
        num_loras=1,
        bin_cost=sum,
        pad_to_multiple_of=64,
    )
    assert len(groups) == 1 and len(groups[0]) == 1
    micro = groups[0][0]
    num, tv = tensors(micro)
    return micro, {"micro": old_fields(micro), "tensor": num}, labels(micro), tv


def codec(samples, micro, frozen_types):
    result = {}
    for name, typ, obj in [("TrainingSample", TrainingSample, samples[0]), ("MicroBatch", MicroBatch, micro)]:
        old = getattr(frozen_types, name)
        fields = {key: getattr(obj, key) for key in old.__struct_fields__}
        old_obj = old(**fields)
        old_bytes = msgspec.msgpack.encode(old_obj)
        decoded = msgspec.msgpack.decode(old_bytes, type=typ)
        assert getattr(decoded, "token_policy_versions", None) is None
        assert old_fields(decoded) == old_fields(old_obj)
        new_none = typ(**fields)
        none_bytes = msgspec.msgpack.encode(new_none)
        old_decoded = msgspec.msgpack.decode(none_bytes, type=old)
        assert old_fields(old_decoded) == old_fields(old_obj)
        result[name] = {
            "old_packet_fields": len(msgspec.msgpack.decode(old_bytes)),
            "new_none_packet_fields": len(msgspec.msgpack.decode(none_bytes)),
            "identical_none_bytes": old_bytes == none_bytes,
            "old_to_new": True,
            "new_none_to_old": True,
        }
        if hasattr(obj, "token_policy_versions") and obj.token_policy_versions is not None:
            populated = msgspec.msgpack.encode(obj)
            roundtrip = msgspec.msgpack.decode(populated, type=typ)
            assert roundtrip == obj
            old_populated = msgspec.msgpack.decode(populated, type=old)
            assert old_fields(old_populated) == old_fields(old_obj)
            result[name]["new_populated_to_old_ignores_trailing"] = True
    required = 5
    minimal = frozen_types.TrainingSample(
        token_ids=[1], mask=[False], logprobs=[0.0], temperatures=[1.0], env_name="minimal"
    )
    omitted = msgspec.msgpack.encode(msgspec.msgpack.decode(msgspec.msgpack.encode(minimal))[:required])
    decoded = msgspec.msgpack.decode(omitted, type=TrainingSample)
    assert old_fields(decoded) == old_fields(minimal) and getattr(decoded, "token_policy_versions", None) is None
    result["omitted_trailing_defaults"] = True
    return result


def boundary_controls(sample):
    """Actual received sample through truncation, distribution dummies, and strict decoder."""
    from prime_rl.trainer.batch import prepare_sample
    from prime_rl.transport.provenance import decode_policy_versions
    from prime_rl.transport.types import EncodedTensor

    original = labels(sample)
    assert original is not None and 5 in original and 6 in original
    prompt_len = original.index(5)
    cut = prompt_len + 3
    truncated = prepare_sample(sample, cut)
    assert truncated.input_ids == sample.token_ids[:cut]
    assert labels(truncated) == original[:cut]
    assert labels(prepare_sample(sample, prompt_len)) is None
    groups = prepare_batch(
        [sample], seq_len=1536, num_train_workers=2, idxs=[0], num_loras=1, bin_cost=sum, pad_to_multiple_of=64
    )
    rows = [row for group in groups for row in group]
    assert len(rows) == 2 and all(len(row.input_ids) % 64 == 0 for row in rows)
    real = [row for row in rows if any(row.loss_mask)]
    dummy = [row for row in rows if not any(row.loss_mask)]
    assert len(real) == len(dummy) == 1
    assert labels(real[0]) == original + [-1] * (len(real[0].input_ids) - len(original))
    assert labels(dummy[0]) is None and not any(dummy[0].advantages)
    assert tensors(dummy[0])[1] is None
    valid = sample.token_policy_versions
    assert not decode_policy_versions(valid, len(sample.token_ids)).flags.writeable
    malformed = {
        "dtype": EncodedTensor(dtype="float64", shape=valid.shape, data=valid.data),
        "shape": EncodedTensor(dtype="int64", shape=[len(original) + 1], data=valid.data),
        "bytes": EncodedTensor(dtype="int64", shape=valid.shape, data=valid.data[:-1]),
        "below_unknown": EncodedTensor(
            dtype="int64", shape=valid.shape, data=np.array([-2] + original[1:], dtype="<i8").tobytes()
        ),
    }
    rejected = []
    for name, payload in malformed.items():
        try:
            decode_policy_versions(payload, len(original))
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(f"accepted malformed {name}")
    return {
        "truncation": labels(truncated),
        "prompt_only_unknown": True,
        "distribution_workers": len(groups),
        "dummy_unknown_zero_loss": True,
        "alignment_multiple": 64,
        "invalid_decoder_rejected": rejected,
    }


def guard(root):
    paths = [
        "src/prime_rl/orchestrator/episodes.py",
        "src/prime_rl/orchestrator/trajectories.py",
        "src/prime_rl/orchestrator/train_sink.py",
        "src/prime_rl/transport/types.py",
        "src/prime_rl/trainer/batch.py",
        "src/prime_rl/trainer/rl/data.py",
        "src/prime_rl/value/credit.py",
        "src/prime_rl/value/trajectory.py",
        "src/prime_rl/orchestrator/algo/grpo.py",
        "src/prime_rl/orchestrator/algo/tether.py",
    ]
    for p in ["src/prime_rl/orchestrator/provenance.py", "src/prime_rl/transport/provenance.py"]:
        if (root / p).exists():
            paths.append(p)
    for module in [
        "prime_rl.orchestrator.episodes",
        "prime_rl.orchestrator.train_sink",
        "prime_rl.trainer.batch",
        "prime_rl.trainer.rl.data",
    ]:
        assert Path(importlib.import_module(module).__file__).resolve().is_relative_to(root)
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    harness_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source = guard(args.source)
    u11 = load_module("unit11", args.source / "experiments/native-producer/acceptance.py")
    dependency_guard = u11.source_guard(args.source)
    frozen_types = load_module("frozen_wire_types", args.baseline / "src/prime_rl/transport/types.py")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    data = RGMixTaskset(RGMixConfig(dataset_path=args.dataset, num_train_examples=1)).load()[0].data
    completions = [
        tokenizer.encode("<answer>" + str(data.scorer_entry["answer"]) + "</answer>", add_special_tokens=False),
        tokenizer.encode("<answer>incorrect</answer>", add_special_tokens=False),
    ]
    tail = tokenizer.encode(" trailing", add_special_tokens=False)
    assert len(tail) == 1
    width = max(map(len, completions)) + 2
    completions = [x + tail * (width - len(x)) + [tokenizer.eos_token_id] for x in completions]
    state = {"tokens": [], "spans": None}
    requests = []

    async def generate(request):
        requests.append(await request.json())
        choice = {
            "token_ids": state["tokens"],
            "finish_reason": "stop",
            "logprobs": {
                "content": [
                    {"token": f"token_id:{t}", "logprob": -0.123456789123 - i * 0.000000001}
                    for i, t in enumerate(state["tokens"])
                ]
            },
        }
        if state["spans"] is not None:
            choice["weight_versions"] = state["spans"]
        return web.json_response({"request_id": "fixed", "choices": [choice]})

    app = web.Application()
    app.router.add_post("/inference/v1/generate", generate)
    app.router.add_get(
        "/v1/models", lambda request: web.json_response({"data": [{"id": "version-model", "max_model_len": 32768}]})
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    config = vf.TrainClientConfig(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key_var="VERSION_FIXTURE_NO_SECRET",
        renderer_model_name=str(args.tokenizer),
        renderer={"name": "qwen3"},
    )
    sampling = {
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 256,
        "extra_body": {"top_k": -1, "min_p": 0, "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0},
    }
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=worker, args=(u11.env_config("rg", "candidate", args.dataset, worker_side=True), queue)
    )
    process.start()
    client = None
    cases = []
    try:
        client = EnvClient(address=await asyncio.wait_for(asyncio.to_thread(queue.get, True, 40), 45))
        await client.wait_for_server_startup(timeout=20)
        modes = (
            ["absent", "clipped_absent"]
            if args.mode == "baseline"
            else ["absent", "mixed", "opaque", "clipped_absent", "clipped", "frozen_absent", "frozen_known"]
        )
        for mode in modes:
            group = []
            episodes = []
            expected = []
            start = len(requests)
            for member, completion in enumerate(completions):
                n = len(completion)
                span_mode = "mixed" if mode in ["clipped", "frozen_known"] else mode
                if span_mode in ["absent", "clipped_absent", "frozen_absent"]:
                    spans = None
                    steps = [-1] * n
                elif span_mode == "opaque":
                    spans = [{"version": "default", "start": 0, "end": n}]
                    steps = [-1] * n
                elif member == 0:
                    spans = [{"version": "5", "start": 0, "end": 2}, {"version": "6", "start": 2, "end": n}]
                    steps = [5] * 2 + [6] * (n - 2)
                else:
                    spans = [{"version": "007", "start": 0, "end": n}]
                    steps = [-1] * n
                state.update(tokens=completion, spans=spans)
                limit = (len(requests[0]["token_ids"]) + 3) if mode in ["clipped", "clipped_absent"] else 32768
                train = u11.training_config("rg", "candidate", args.dataset, limit)
                env = u11.RecordedEnv(train.train.env[0])
                env._env_client = client
                env.sampling_args = sampling
                r = await asyncio.wait_for(
                    env.run_rollout(config, 0, "version-model", None, task_data=data.model_dump(mode="python")), 40
                )
                assert r.ok and env.episode.model_dump(mode="python") == env.before
                physical = r.source_physical_trace
                assert physical is not None and [
                    s.model_dump(mode="python") for node in physical.nodes for s in node.weight_versions or []
                ] == (spans or [])
                (args.output / f"episode-{mode}-{member}.json").write_text(json.dumps(env.before, indent=2) + "\n")
                prompt = requests[-1]["token_ids"]
                full = prompt + completion
                assert [t for node in physical.nodes for t in node.token_ids] == full
                kept = min(limit, len(full))
                expected.append(([-1] * len(prompt) + steps)[:kept])
                group.append(r)
                episodes.append(env.before)
            contexts = {}
            saved_samples = None
            saved_micro = None
            for context_name in ["task", "group_leave_one_out"]:
                train = u11.training_config("rg", "candidate", args.dataset, limit, context_name)
                if mode.startswith("frozen"):
                    raw = train.model_dump(mode="python")
                    raw["train"]["env"][0]["algo"] = {
                        "type": "sft",
                        "sampling": {"source": {"name": "frozen-model", "base_url": [f"http://127.0.0.1:{port}/v1"]}},
                    }
                    train = type(train).model_validate(raw)
                samples, numeric, carrier, live = await finalize(group, train, tokenizer)
                wanted = [None if not live or all(x == -1 for x in row) else row for row in expected]
                assert carrier == wanted, (mode, carrier, wanted)
                micro, packed, packed_versions, tensor_versions = pack(samples)
                if any(x is not None for x in wanted):
                    joined = [x for row in expected for x in row] + [-1] * (
                        len(micro.input_ids) - sum(map(len, expected))
                    )
                    assert packed_versions == joined and tensor_versions == [joined]
                else:
                    assert packed_versions is tensor_versions is None
                contexts[context_name] = {
                    "numeric": numeric,
                    "packed": packed,
                    "sample_versions": carrier,
                    "packed_versions": packed_versions,
                    "tensor_versions": tensor_versions,
                    "live_policy": live,
                }
                saved_samples, saved_micro = samples, micro
            case = {"mode": mode, "contexts": contexts, "requests": requests[start:], "source_episodes": episodes}
            case["codec"] = codec(saved_samples, saved_micro, frozen_types)
            if mode == "mixed":
                case["boundary_controls"] = boundary_controls(saved_samples[0])
            (args.output / f"{mode}-sample.msgpack").write_bytes(msgspec.msgpack.encode(saved_samples[0]))
            (args.output / f"{mode}-micro.msgpack").write_bytes(msgspec.msgpack.encode(saved_micro))
            cases.append(case)
            (args.output / "raw.json").write_text(json.dumps({"cases": cases, "requests": requests}, indent=2) + "\n")
        assert guard(args.source) == source
        assert u11.source_guard(args.source) == dependency_guard
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == harness_sha
        result = {
            "status": "passed",
            "mode": args.mode,
            "source_hashes": source,
            "harness_sha256": harness_sha,
            "dependency_provenance": dependency_guard,
            "case_count": len(cases),
            "http_requests": len(requests),
            "worker_pid": process.pid,
            "raw_sha256": hashlib.sha256((args.output / "raw.json").read_bytes()).hexdigest(),
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        if client is not None:
            await client.close()
        if process.is_alive():
            process.terminate()
        await asyncio.to_thread(process.join, 10)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        assert not process.is_alive()
        await runner.cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "candidate"], required=True)
    for name in ["source", "baseline", "dataset", "tokenizer", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.source = args.source.resolve()
    asyncio.run(main(args))
