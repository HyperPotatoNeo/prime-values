"""Read-only same-response legacy/native geometry audit; no model or GPU use."""

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import verifiers.v1 as vf
from aiohttp import web
from rg_mix_v1.taskset import RGMixConfig, RGMixTaskset
from transformers import AutoTokenizer
from verifiers.v1.legacy import LegacyEnvServer
from verifiers.v1.serve import EnvClient, EnvServer

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.orchestrator.episodes import from_episode
from prime_rl.orchestrator.trajectories import materialize_training_data
from prime_rl.value.credit import compute_value_result
from prime_rl.value.trajectory import snapshot_value_trajectory


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def project(episode):
    assert episode.ok and len(episode.traces) == 1
    rollout = from_episode(episode.model_copy(deep=True))
    materialized = materialize_training_data(rollout, env_name="boundary", capture_sampled_masks=True)
    owned = snapshot_value_trajectory(
        materialized.samples,
        sampled_masks=materialized.sampled_masks,
        layout=materialized.layout,
        reward=rollout.reward,
    )
    config = ValueFunctionConfig(
        model={"name": "deterministic-zero-service", "seq_len": 32768},
        gamma=0.9,
        gae_lambda=1.0,
        value_target_lambda=1.0,
    )
    predictions = [[0.0] * len(sample.token_ids) for sample in owned.samples]
    values, advantages, targets = compute_value_result(owned, predictions, config, layout=None)
    action_indices = [[i for i, mask in enumerate(sample.mask) if mask] for sample in owned.samples]
    action_targets = [[target[i] for i in indices] for target, indices in zip(targets, action_indices, strict=True)]
    for sequence in action_targets:
        assert sequence[-1] == rollout.reward == 1.0
        for i, target in enumerate(sequence):
            assert abs(target - 0.9 ** (len(sequence) - i - 1)) < 1e-14
    return {
        "reward": rollout.reward,
        "metrics": dict(rollout.metrics),
        "is_completed": rollout.is_completed,
        "is_truncated": rollout.is_truncated,
        "stop_condition": rollout.stop_condition,
        "num_output_tokens": rollout.num_output_tokens,
        "sample_lengths": [len(sample.token_ids) for sample in owned.samples],
        "action_counts": [len(indices) for indices in action_indices],
        "terminal_action_indices": [indices[-1] for indices in action_indices],
        "action_targets": action_targets,
        "owned": asdict(owned),
        "values": values,
        "advantages": advantages,
        "targets": targets,
    }


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    acceptance = load_module("frozen_unit9_acceptance", args.source / "experiments/native-rg/acceptance.py")
    provenance = acceptance.source_provenance(args.source)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    task = RGMixTaskset(RGMixConfig(dataset_path=args.dataset)).load()[0]
    # Real cached answer with a trailing explanation yields correct reward while
    # leaving enough sampled tokens to cross a deliberately small actor budget.
    completion = "<answer>" + str(task.data.scorer_entry["answer"]) + "</answer>" + " trailing" * 20
    response_ids = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
    assert len(response_ids) > 6
    requests = []

    async def generate(request):
        requests.append(await request.json())
        return web.json_response(
            {
                "request_id": "fixed-sequence-boundary",
                "choices": [
                    {
                        "token_ids": response_ids,
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"logprob": -0.4 - i * 0.001} for i in range(len(response_ids))]},
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
    sampling = vf.SamplingConfig(
        temperature=1,
        top_p=1,
        max_tokens=256,
        extra_body={
            "top_k": -1,
            "min_p": 0,
            "repetition_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )
    load_module("frozen_boundary_rg", args.legacy)
    legacy = LegacyEnvServer(
        "frozen_boundary_rg",
        {
            "dataset_path": str(args.dataset),
            "num_train_examples": 10000,
            "num_eval_examples": 256,
        },
        address="tcp://127.0.0.1:0",
        extra_env_kwargs={"max_seq_len": 32768},
    )
    native = EnvServer(
        EnvConfig.model_validate(
            {
                "env": {
                    "taskset": {
                        "id": "rg-mix-v1",
                        "dataset_path": "unused-worker-path",
                    },
                    "agent": {
                        "harness": {"id": "null", "runtime": {"type": "subprocess"}},
                        "timeout": {"rollout": 60},
                    },
                }
            }
        ).env,
        address="tcp://127.0.0.1:0",
    )
    servers = [asyncio.create_task(server.run()) for server in (legacy, native)]
    clients = [EnvClient(address=server.address) for server in (legacy, native)]
    try:
        for client in clients:
            await client.wait_for_server_startup()
        native_episode = await clients[1].run(
            task_data=task.data.model_dump(mode="json"),
            client=client_config,
            model="boundary-model",
            sampling=sampling,
        )
        native_projection = project(native_episode)
        prompt_length = len(requests[0]["token_ids"])
        total = prompt_length + len(response_ids)
        assert native_projection["sample_lengths"] == [total]
        cases = []
        for name, limit in (
            ("within_budget", total + 1),
            ("completion_crosses_limit", prompt_length + 3),
            ("prompt_crosses_limit", prompt_length - 1),
        ):
            legacy.env.set_max_seq_len(limit)
            legacy_episode = await clients[0].run(
                task_idx=0,
                client=client_config,
                model="boundary-model",
                sampling=sampling,
            )
            old = project(legacy_episode)
            assert requests[-1] == requests[0], "The endpoint received different requests"
            assert old["reward"] == native_projection["reward"] == 1
            if name == "within_budget":
                assert old["owned"] == native_projection["owned"]
                assert old["targets"] == native_projection["targets"]
                assert old["is_truncated"] == native_projection["is_truncated"] is False
            elif name == "completion_crosses_limit":
                assert old["sample_lengths"] == [limit] and old["action_counts"] == [3]
                assert old["action_targets"][0][0] == 0.81
                assert old["action_targets"][0] != native_projection["action_targets"][0][:3]
            else:
                assert old["sample_lengths"] == [] and old["action_counts"] == []
            cases.append(
                {
                    "case": name,
                    "actor_seq_len": limit,
                    "legacy": old,
                    "native": native_projection,
                }
            )
        report = {
            "status": "reproduced",
            "cases": cases,
            "requests_identical": True,
            "prompt_tokens": prompt_length,
            "completion_tokens": len(response_ids),
            "critic_seq_len": 32768,
            "gamma": 0.9,
            "gae_lambda": 1.0,
            "value_target_lambda": 1.0,
            "provenance": provenance,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "legacy_source_sha256": hashlib.sha256(args.legacy.read_bytes()).hexdigest(),
            "scope": "Actual pinned legacy/native env servers and training renderer, exact cached RG task, same deterministic HTTP response. Real canonical materializer/snapshot/credit math, zero-value service inputs. No model, GPU, TrainSink publication or proposed behavior change.",
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "requests.json").write_text(json.dumps(requests, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "prompt_tokens": prompt_length,
                    "completion_tokens": len(response_ids),
                    "cases": [
                        {
                            "case": c["case"],
                            "legacy_actions": c["legacy"]["action_counts"],
                            "native_actions": c["native"]["action_counts"],
                            "legacy_truncated": c["legacy"]["is_truncated"],
                            "native_truncated": c["native"]["is_truncated"],
                        }
                        for c in cases
                    ],
                }
            )
        )
    finally:
        for client in clients:
            await client.close()
        for server in servers:
            server.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server
        await http.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "dataset", "tokenizer", "legacy", "output"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    asyncio.run(asyncio.wait_for(main(parser.parse_args()), timeout=180))
