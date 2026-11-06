"""Real legacy/native producer and training-projection contract, without a model.

Pinned tokenizer and actual cached RG scorers run against deterministic token
responses. Actual Env.run_rollout, TrainSink, value credit and policy filters run.
The original Episode is retained and checked; no GPU/inference-kernel claim.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import msgspec
import verifiers.v1 as vf
from aiohttp import web
from renderers.qwen3 import Qwen3Renderer
from rg_mix_v1.taskset import RGMixConfig, RGMixTaskset
from transformers import AutoTokenizer
from verifiers.v1.legacy import LegacyEnvServer
from verifiers.v1.serve import EnvClient, EnvServer

from prime_rl.configs.orchestrator import EnvConfig, OrchestratorConfig
from prime_rl.orchestrator.envs import Env, TrainEnvs
from prime_rl.orchestrator.filters import GibberishFilter
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.trajectories import materialize_training_data


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config_for(dataset, limit, *, projection=False, context="task"):
    env = {
        "name": "rg-boundary",
        "env": {"taskset": {"id": "rg-mix-v1", "dataset_path": str(dataset)}},
        "algo": {"type": "grpo", "baseline": {"type": "value"}, "branch_semantics": "sequential"},
    }
    if projection:
        env["training_projection"] = {"type": "single_turn"}
    config = OrchestratorConfig(
        model={"name": "Qwen/Qwen3-4B-Instruct-2507"},
        seq_len=limit,
        batch_size=2,
        group_size=2,
        max_inflight_rollouts=4,
        train={"env": [env]},
        value_function={
            "model": {"name": "deterministic-service", "seq_len": 32768},
            "gamma": 0.9,
            "gae_lambda": 0.35,
            "value_target_lambda": 0.8,
            "privileged_context": context,
        },
    )
    assert OrchestratorConfig.model_validate_json(config.model_dump_json()) == config
    if projection:
        assert config.train.env[0].training_projection.max_seq_len == limit
    return config


class RecordedEnv(Env):
    async def run_episode(self, *args, **kwargs):
        self.episode = await super().run_episode(*args, **kwargs)
        self.before = self.episode.model_dump(mode="json")
        return self.episode


def projection_record(rollout):
    data = materialize_training_data(rollout, env_name="rg-boundary", capture_sampled_masks=True)
    return {
        "reward": rollout.reward,
        "metrics": {k: rollout.metrics[k] for k in ("check_answer_reward_func", "format_reward_func")},
        "samples": [
            {"token_ids": list(s.token_ids), "mask": list(s.mask), "logprobs": list(s.logprobs)} for s in data.samples
        ],
        "sampled_masks": [list(m) for m in data.sampled_masks],
        "is_completed": rollout.is_completed,
        "is_truncated": rollout.is_truncated,
        "num_output_tokens": rollout.num_output_tokens,
        "num_total_tokens": rollout.num_total_tokens,
        "usage": rollout.usage.model_dump(mode="json"),
        "call_finish_reasons": [c.finish_reason for c in rollout.calls],
        "messages": [n.message.model_dump(mode="json") for n in rollout.nodes],
        "gibberish": GibberishFilter("tail", token_id_threshold=0, logprob_threshold=-2, enforce=True)
        .check(rollout)
        .detected,
    }


async def sink_record(originals, tokenizer, dataset, limit, context):
    requests, published, prompts = [], [], []

    class Evaluator:
        async def evaluate(self, token_ids):
            requests.append(token_ids)
            return SimpleNamespace(values=[[0.0] + [0.01 * (t % 11) for t in row[1:]] for row in token_ids], version=7)

    class Renderer:
        def render_ids(self, messages, *, add_generation_prompt):
            prompts.append(messages[0]["content"])
            return Qwen3Renderer(tokenizer).render_ids(messages, add_generation_prompt=add_generation_prompt)

    config = config_for(dataset, limit, context=context)
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
        pre_filters=[GibberishFilter("tail", token_id_threshold=0, logprob_threshold=-2, enforce=True)],
        post_filters=[],
        value_publisher=SimpleNamespace(publish=published.append),
    )
    rollouts = [r.model_copy(deep=True) for r in originals]
    try:
        for r in rollouts:
            r.env_name = "rg-boundary"
            r.group_id = uuid.UUID(int=1)
            r.policy_version = 3
        assert await sink.add(rollouts[0]) is None
        if id(rollouts[0]) in sink.scoring_tasks:
            await sink.scoring_tasks[id(rollouts[0])]
        assert published == [] and prompts == []
        if context == "group_leave_one_out":
            assert requests == []
        batch = await sink.add(rollouts[1])
        result = {
            "evaluator_requests": requests,
            "prefix_prompts": prompts,
            "published": [[msgspec.to_builtins(s) for s in item.samples] for item in published],
            "policy_batch_samples": [] if batch is None else [msgspec.to_builtins(s) for s in batch.samples],
            "rollouts": [
                {
                    "reward": r.reward,
                    "values": r.value_predictions,
                    "advantages": r.value_advantages,
                    "targets": r.value_returns,
                    "wire_advantages": [s.advantages for s in r.samples],
                    "filtered": r.is_filtered,
                    "filter_results": r.filter_results,
                }
                for r in rollouts
            ],
        }
        before = json.loads(json.dumps(result))
        for r in rollouts:
            r.nodes.clear()
            r.samples.clear()
        assert result == before, "Saved sink evidence aliased released policy data"
        assert [[msgspec.to_builtins(s) for s in item.samples] for item in published] == result["published"]
        return result
    finally:
        await sink.stop()


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    unit9 = load_module("unit9_acceptance", args.source / "experiments/native-rg/acceptance.py")
    provenance = unit9.source_provenance(args.source)
    # The unit9 guard deliberately checks unmodified verifier/renderer/package
    # modules too. Add every production file changed by the projection itself.
    for rel in [
        "src/prime_rl/orchestrator/episodes.py",
        "src/prime_rl/orchestrator/types.py",
        "src/prime_rl/orchestrator/envs.py",
        "packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py",
    ]:
        provenance["source_sha256"][rel] = hashlib.sha256((args.source / rel).read_bytes()).hexdigest()
    import prime_rl.configs.orchestrator as config_module
    import prime_rl.orchestrator.envs as env_module
    import prime_rl.orchestrator.types as types_module

    for module, relative in [
        (env_module, "src/prime_rl/orchestrator/envs.py"),
        (types_module, "src/prime_rl/orchestrator/types.py"),
    ]:
        assert Path(module.__file__).resolve() == (args.source / relative).resolve()

    assert (
        Path(config_module.__file__).resolve()
        == (args.source / "packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py").resolve()
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    task = RGMixTaskset(RGMixConfig(dataset_path=args.dataset, num_train_examples=1)).load()[0]
    correct = tokenizer.encode(
        "<answer>" + str(task.data.scorer_entry["answer"]) + "</answer>", add_special_tokens=False
    )
    incorrect = tokenizer.encode("<answer>incorrect</answer>", add_special_tokens=False)
    tail = tokenizer.encode(" trailing", add_special_tokens=False)
    assert len(tail) == 1
    width = max(len(correct), len(incorrect)) + 20
    responses = [ids + tail * (width - len(ids)) + [tokenizer.eos_token_id] for ids in (correct, incorrect)]
    current_ids = responses[0]
    requests = []

    async def generate(request):
        requests.append(await request.json())
        logprobs = [-0.4] * len(current_ids)
        logprobs[-2] = -5.0
        return web.json_response(
            {
                "request_id": "fixed-boundary",
                "choices": [
                    {
                        "token_ids": current_ids,
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"logprob": x} for x in logprobs]},
                    }
                ],
            }
        )

    async def models(request):
        return web.json_response({"data": [{"id": "boundary-model", "max_model_len": 32768}]})

    app = web.Application()
    app.router.add_post("/inference/v1/generate", generate)
    app.router.add_get("/v1/models", models)
    http = web.AppRunner(app)
    await http.setup()
    site = web.TCPSite(http, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client_config = vf.TrainClientConfig(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key_var="BOUNDARY_NO_SECRET",
        renderer_model_name=str(args.tokenizer),
        renderer={"name": "qwen3"},
    )
    sampling = {
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 256,
        "extra_body": {"top_k": -1, "min_p": 0, "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0},
    }
    load_module("frozen_projection_rg", args.legacy)
    legacy = LegacyEnvServer(
        "frozen_projection_rg",
        {"dataset_path": str(args.dataset), "num_train_examples": 10000, "num_eval_examples": 256},
        address="tcp://127.0.0.1:0",
        extra_env_kwargs={"max_seq_len": 32768},
    )
    native = EnvServer(
        EnvConfig.model_validate(
            {
                "env": {
                    "taskset": {"id": "rg-mix-v1", "dataset_path": "missing-worker-dataset"},
                    "agent": {"harness": {"id": "null", "runtime": {"type": "subprocess"}}, "timeout": {"rollout": 60}},
                }
            }
        ).env,
        address="tcp://127.0.0.1:0",
    )
    servers = [asyncio.create_task(s.run()) for s in (legacy, native)]
    clients = [EnvClient(address=s.address) for s in (legacy, native)]

    async def produce(config, client, payload):
        env = RecordedEnv(config)
        env._env_client = client
        env.sampling_args = sampling
        r = await env.run_rollout(client_config, 0, "boundary-model", None, task_data=payload)
        assert env.episode.model_dump(mode="json") == env.before, "Projection mutated the original Episode"
        if r.source_physical_trace is not None:
            assert r.source_physical_trace is env.episode.traces[0]
            assert r.source_physical_trace.model_dump(mode="json") == env.before["traces"][0]
        assert "source_physical_trace" not in r.model_dump(mode="json")
        return r, env.before

    try:
        for c in clients:
            await c.wait_for_server_startup()
        native_full, physical = [], []
        for current_ids in responses:
            r, record = await produce(
                config_for(args.dataset, 32768).train.env[0], clients[1], task.data.model_dump(mode="json")
            )
            native_full.append(r)
            physical.append(record)
        prompt_length = len(requests[0]["token_ids"])
        full_length = prompt_length + len(responses[0])
        assert len(responses[0]) == len(responses[1])
        assert [r.reward for r in native_full] == [1, 0]
        native_records = [projection_record(r) for r in native_full]
        for x in native_records:
            assert x["gibberish"] and not x["is_truncated"]
        native_sink = {
            context: await sink_record(native_full, tokenizer, args.dataset, full_length + 1, context)
            for context in ("task", "group_leave_one_out")
        }
        cases = []
        for name, limit in [
            ("within", full_length + 1),
            ("exact", full_length),
            ("completion_clipped", prompt_length + 3),
            ("prompt_overflow", prompt_length - 1),
        ]:
            legacy.env.set_max_seq_len(limit)
            old_group, projected_group = [], []
            raw_pairs = []
            for member, current_ids in enumerate(responses):
                start = len(requests)
                old, old_raw = await produce(EnvConfig(id="frozen_projection_rg"), clients[0], None)
                projected, original = await produce(
                    config_for(args.dataset, limit, projection=True).train.env[0],
                    clients[1],
                    task.data.model_dump(mode="json"),
                )
                assert requests[start] == requests[start + 1] == requests[0]
                old_record, new_record = projection_record(old), projection_record(projected)
                (args.output / f"pair-{name}-{member}.json").write_text(
                    json.dumps({"legacy": old_record, "projected": new_record}, indent=2) + "\n"
                )
                # The frozen bridge fills missing reasoning usage with zero;
                # native preserves unknown=None. Preserve/report both raw values.
                assert old_record["usage"]["reasoning_tokens"] == 0
                assert new_record["usage"]["reasoning_tokens"] is None
                old_compare = json.loads(json.dumps(old_record))
                new_compare = json.loads(json.dumps(new_record))
                del old_compare["usage"]["reasoning_tokens"]
                del new_compare["usage"]["reasoning_tokens"]
                assert old_compare == new_compare, ("projected geometry/metadata differs", name, member)
                assert new_record["usage"] == native_records[member]["usage"] and new_record["call_finish_reasons"] == [
                    "stop"
                ]
                marker = projected.info["prime_rl_training_projection"]
                assert marker["max_seq_len"] == limit and marker["physical_tokens"] == full_length
                assert marker["clipped"] == (full_length > limit)
                if full_length > limit:
                    assert (
                        projected.stop_condition == "max_total_tokens"
                        and projected.source_physical_trace.is_truncated is False
                    )
                raw_pairs.append(
                    {
                        "member": member,
                        "requests": requests[start : start + 2],
                        "legacy": old_record,
                        "projected_native": new_record,
                        "unprojected_native": native_records[member],
                        "projection_receipt": marker,
                        "legacy_stop_condition": old.stop_condition,
                        "projected_stop_condition": projected.stop_condition,
                        "source_physical_episode": original,
                    }
                )
                old_group.append(old)
                projected_group.append(projected)
            contexts = {}
            for context in ["task", "group_leave_one_out"]:
                old_sink = await sink_record(old_group, tokenizer, args.dataset, limit, context)
                projected_sink = await sink_record(projected_group, tokenizer, args.dataset, limit, context)
                assert old_sink == projected_sink, ("sink differs", name, context)
                if name in ("within", "exact"):
                    assert old_sink == native_sink[context]
                elif name == "completion_clipped":
                    assert old_sink["rollouts"][0]["targets"] != native_sink[context]["rollouts"][0]["targets"]
                    assert all(r["filtered"] for r in native_sink[context]["rollouts"])
                    assert len(native_sink[context]["published"]) == 2
                if name == "prompt_overflow":
                    assert (
                        not old_sink["published"]
                        and not old_sink["evaluator_requests"]
                        and not old_sink["policy_batch_samples"]
                    )
                elif name == "completion_clipped":
                    assert (
                        len(old_sink["published"]) == 2
                        and len(old_sink["policy_batch_samples"]) == 2
                        and not any(r["filtered"] for r in old_sink["rollouts"])
                    )
                else:
                    assert (
                        len(old_sink["published"]) == 2
                        and not old_sink["policy_batch_samples"]
                        and all(r["filtered"] for r in old_sink["rollouts"])
                    )
                contexts[context] = {"legacy": old_sink, "projected_native": projected_sink}
            cases.append({"case": name, "actor_seq_len": limit, "pairs": raw_pairs, "sink": contexts})
        raw = {
            "cases": cases,
            "unprojected_physical_episodes": physical,
            "unprojected_native_sink": native_sink,
            "all_requests": requests,
        }
        (args.output / "paired-raw.json").write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
        result = {
            "status": "passed",
            "cases": [
                {
                    "case": c["case"],
                    "limit": c["actor_seq_len"],
                    "retained_actions": [sum(s["mask"]) for s in c["pairs"][0]["projected_native"]["samples"]],
                    "raw_usage": c["pairs"][0]["projected_native"]["usage"],
                }
                for c in cases
            ],
            "http_requests": len(requests),
            "prompt_tokens": prompt_length,
            "completion_tokens": len(responses[0]),
            "provenance": provenance,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "paired_raw_sha256": hashlib.sha256((args.output / "paired-raw.json").read_bytes()).hexdigest(),
            "raw_episode_unchanged": True,
            "unprojected_native_sink_baseline": True,
            "preexisting_usage_difference": "Legacy reasoning_tokens=0 versus native unknown=None; explicitly asserted and raw values retained. Projected native preserves complete original native Usage exactly.",
            "named_filter_tail_control": "Unprojected bad tail detected; clipped tail absent from both actual canonical branch filters and emitted policy samples; critic publishes before filtering.",
            "scope": "Actual fixed-response producer/renderer/scorer, Env.run_rollout configprojection, TrainSink task/LOO credit/publication and policyfilter; no model or GPU.",
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": "passed", "cases": result["cases"], "requests": len(requests)}))
    finally:
        for c in clients:
            await c.close()
        for t in servers:
            t.cancel()
        await asyncio.gather(*servers)
        await http.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ["source", "dataset", "tokenizer", "legacy", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    for key in ["source", "dataset", "tokenizer", "legacy", "output"]:
        setattr(args, key, getattr(args, key).resolve())
    asyncio.run(asyncio.wait_for(main(args), timeout=300))
