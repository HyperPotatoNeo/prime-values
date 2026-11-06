"""Actual two-worker RG transport/renderer/value-materialization acceptance, CPU only.

The HTTP fixture supplies deterministic completion token IDs; it does not run a
model or claim inference-kernel parity. Both production training clients perform
real pinned-tokenizer rendering and real RG scoring over the frozen saved cache.
"""

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import msgspec
import verifiers.v1 as vf
from aiohttp import web
from rg_mix_v1.taskset import RGMixConfig, RGMixTaskset
from transformers import AutoTokenizer
from verifiers.v1.legacy import LegacyEnvServer
from verifiers.v1.serve import EnvClient, EnvServer

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.orchestrator.episodes import from_episode
from prime_rl.orchestrator.trajectories import materialize_training_data
from prime_rl.value.trajectory import snapshot_value_trajectory


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def worker(raw, queue, audit_path):
    """Only instrument task reconstruction; execution remains the actual EnvServer."""

    class AuditedServer(EnvServer):
        def _build_task(self, data):
            task = super()._build_task(data)
            with open(audit_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "idx": task.data.idx,
                            "payload_sha256": digest(task.data.model_dump(mode="json")),
                        }
                    )
                    + "\n"
                )
            return task

    server = AuditedServer(EnvConfig.model_validate(raw).env, address="tcp://127.0.0.1:0")
    queue.put(server.address)
    asyncio.run(server.run())


def project(episode, idx):
    assert episode.ok and len(episode.traces) == 1, episode.model_dump(mode="json")
    rollout = from_episode(episode.model_copy(deep=True), task_idx=idx)
    assert rollout.ok and rollout.num_turns == 1
    materialized = materialize_training_data(rollout, env_name="rg-parity", capture_sampled_masks=True)
    assert len(materialized.samples) == 1 and any(materialized.samples[0].mask)
    owned = snapshot_value_trajectory(
        materialized.samples,
        sampled_masks=materialized.sampled_masks,
        layout=materialized.layout,
        reward=rollout.reward,
        env_name="rg-parity",
        group_id="fixed",
        policy_version=3,
    )
    result = {
        "reward": rollout.reward,
        "metrics": {name: rollout.metrics[name] for name in ("check_answer_reward_func", "format_reward_func")},
        "samples": [
            {"token_ids": list(s.token_ids), "mask": list(s.mask), "logprobs": list(s.logprobs)}
            for s in materialized.samples
        ],
        "owned": asdict(owned),
        "turns": rollout.num_turns,
        "is_completed": rollout.is_completed,
        "is_truncated": rollout.is_truncated,
        "completion_len": rollout.num_output_tokens,
    }
    assert len(result["samples"][0]["token_ids"]) > 0 and sum(result["samples"][0]["mask"]) > 0
    frozen_owned = asdict(owned)
    materialized.samples[0].token_ids.clear()
    materialized.samples[0].mask.clear()
    rollout.nodes.clear()
    rollout.rewards.clear()
    assert asdict(owned) == frozen_owned, "Value input changed after policy release"
    assert owned.prefix is None, "RG private scoring data entered value prefix"
    return result


async def credit_and_peer_context(groups, tokenizer):
    """Actual sink arrival/group/publication for one mixed-reward cohort."""
    import uuid

    from renderers.qwen3 import Qwen3Renderer

    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.orchestrator.envs import TrainEnvs
    from prime_rl.orchestrator.train_sink import TrainSink
    from prime_rl.orchestrator.value_context import group_leave_one_out_prompts

    outcomes = {}
    for context in ("task", "group_leave_one_out"):
        records = []
        for episodes in groups:
            requests, published, rendered_prompts = [], [], []

            class Evaluator:
                async def evaluate(self, token_ids):
                    requests.append(token_ids)
                    return SimpleNamespace(
                        values=[[0.0] + [0.01 * (token % 11) for token in tokens[1:]] for tokens in token_ids],
                        version=7,
                    )

            class Renderer:
                def render_ids(self, messages, *, add_generation_prompt):
                    rendered_prompts.append(messages[0]["content"])
                    return Qwen3Renderer(tokenizer).render_ids(messages, add_generation_prompt=add_generation_prompt)

            config = OrchestratorConfig(
                model={"name": "Qwen/Qwen3-4B-Instruct-2507"},
                batch_size=2,
                group_size=2,
                max_inflight_rollouts=4,
                train={
                    "env": [
                        {
                            "name": "rg-parity",
                            "env": {"taskset": {"id": "rg-mix-v1", "dataset_path": "unused-worker-path"}},
                            "algo": {"type": "grpo", "baseline": {"type": "value"}, "branch_semantics": "sequential"},
                        }
                    ]
                },
                value_function={
                    "model": {"name": "deterministic-value-service", "seq_len": 32768},
                    "gamma": 0.9,
                    "gae_lambda": 0.35,
                    "value_target_lambda": 0.8,
                    "loss": {"type": "mse"},
                    "privileged_context": context,
                },
            )
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
            rollouts = []
            try:
                for episode in episodes:
                    rollout = from_episode(episode.model_copy(deep=True))
                    rollout.env_name = "rg-parity"
                    rollout.group_id = uuid.UUID(int=1)
                    rollout.policy_version = 3
                    rollouts.append(rollout)
                assert [r.reward for r in rollouts] == [0, 1]
                assert await sink.add(rollouts[0]) is None
                await sink.scoring_tasks[id(rollouts[0])]
                assert rollouts[0].value_prefix is None
                assert rendered_prompts == [] and published == []
                if context == "group_leave_one_out":
                    assert requests == []
                batch = await sink.add(rollouts[1])
                assert batch is not None and len(batch.samples) == 2 and len(published) == 2
                prompts = group_leave_one_out_prompts(rollouts, special_tokens=tokenizer.all_special_tokens)
                for i, prompt in enumerate(prompts):
                    payload = json.loads(
                        prompt.split("BEGIN_PRIVILEGED_PEER_DATA\n")[1].split("\nEND_PRIVILEGED_PEER_DATA")[0]
                    )
                    assert payload["group_size"] == 2 and len(payload["peers"]) == 1
                    assert payload["peers"][0]["reward"] == rollouts[1 - i].reward
                    assert "scorer_entry" not in prompt and "variant_id" not in prompt and "entry_map" not in prompt
                if context == "task":
                    assert rendered_prompts == [] and all(r.value_prefix is None for r in rollouts)
                else:
                    assert rendered_prompts == prompts and all(r.value_prefix is not None for r in rollouts)
                records.append(
                    {
                        "loo_prompts": prompts,
                        "rendered_prefix_prompts": rendered_prompts,
                        "evaluator_requests": requests,
                        "published_samples": [
                            [msgspec.to_builtins(sample) for sample in item.samples] for item in published
                        ],
                        "rollouts": [
                            {
                                "reward": r.reward,
                                "values": r.value_predictions,
                                "advantages": r.value_advantages,
                                "targets": r.value_returns,
                                "wire_advantages": [sample.advantages for sample in r.samples],
                            }
                            for r in rollouts
                        ],
                    }
                )
                immutable = records[-1]["published_samples"]
                for rollout in rollouts:
                    rollout.nodes.clear()
                    rollout.samples.clear()
                assert [[msgspec.to_builtins(sample) for sample in item.samples] for item in published] == immutable
            finally:
                await sink.stop()
        assert records[0] == records[1] == records[2], (
            "Legacy/native sink credit or LOO prompt bytes changed",
            context,
        )
        assert records[0]["rollouts"][0]["targets"] != records[0]["rollouts"][1]["targets"]
        outcomes[context] = {"legacy": records[0], "native_worker_0": records[1], "native_worker_1": records[2]}
    return outcomes


def source_provenance(root):
    import importlib
    import importlib.metadata as metadata

    modules = {
        "prime_rl.orchestrator.episodes": "src/prime_rl/orchestrator/episodes.py",
        "prime_rl.orchestrator.trajectories": "src/prime_rl/orchestrator/trajectories.py",
        "prime_rl.value.trajectory": "src/prime_rl/value/trajectory.py",
        "prime_rl.orchestrator.train_sink": "src/prime_rl/orchestrator/train_sink.py",
        "prime_rl.orchestrator.algo.base": "src/prime_rl/orchestrator/algo/base.py",
        "prime_rl.orchestrator.algo.grpo": "src/prime_rl/orchestrator/algo/grpo.py",
        "prime_rl.value.credit": "src/prime_rl/value/credit.py",
        "prime_rl.orchestrator.value_context": "src/prime_rl/orchestrator/value_context.py",
        "rg_mix_v1.taskset": "packages/rg-mix-env/rg_mix_v1/taskset.py",
        "rg_mix_v1.scoring": "packages/rg-mix-env/rg_mix_v1/scoring.py",
        "rg_mix_shared": "packages/rg-mix-env/rg_mix_shared.py",
        "renderers.client": "deps/renderers/renderers/client.py",
        "verifiers.v1.clients.train": "deps/verifiers/verifiers/v1/clients/train.py",
        "verifiers.v1.legacy": "deps/verifiers/verifiers/v1/legacy.py",
        "verifiers.v1.serve.server": "deps/verifiers/verifiers/v1/serve/server.py",
    }
    hashes = {}
    for name, relative in modules.items():
        path = Path(importlib.import_module(name).__file__)
        assert path.read_bytes() == (root / relative).read_bytes(), (name, path)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    verified = 0
    distribution = metadata.distribution("verifiers")
    for entry in distribution.files:
        relative = str(entry)
        if relative.startswith("verifiers/") and relative.endswith(".py"):
            assert distribution.locate_file(entry).read_bytes() == (root / "deps/verifiers" / relative).read_bytes(), (
                relative
            )
            verified += 1
    assert verified > 100
    return {
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": hashes,
        "verifier_python_files_exact_to_pinned_source": verified,
        "versions": {
            key: metadata.version(key)
            for key in ("torch", "transformers", "verifiers", "renderers", "reasoning-gym", "rg-mix-env")
        },
    }


async def main(args):
    provenance = source_provenance(Path(__file__).resolve().parents[2])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), local_files_only=True)
    source_dataset_hashes = {
        str(p.relative_to(args.dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            args.dataset / "metadata.json",
            args.dataset / "dataset/data-00000-of-00001.arrow",
            args.dataset / "dataset/dataset_info.json",
            args.dataset / "dataset/state.json",
        )
    }
    tasks = RGMixTaskset(RGMixConfig(dataset_path=args.dataset)).load()
    eval_tasks = RGMixTaskset(RGMixConfig(dataset_path=args.dataset, split="eval", num_eval_examples=256)).load()
    selected = {}
    for task in tasks:
        selected.setdefault(task.data.variant_id, task)
    chosen = list(selected.values()) + [tasks[-1], eval_tasks[0], eval_tasks[-1]]
    assert len(tasks) == 10000 and len(eval_tasks) == 256 and len(selected) == 5
    assert [eval_tasks[0].data.idx, eval_tasks[-1].data.idx] == [0, 255]
    # Client-owned selected data is loaded above. Worker config points at an
    # absent dataset: any server-side reload/regeneration fails this real run.
    missing = output / "never-created-worker-dataset"
    raw = {
        "env": {
            "taskset": {"id": "rg-mix-v1", "dataset_path": str(missing)},
            "agent": {"harness": {"id": "null", "runtime": {"type": "subprocess"}}, "timeout": {"rollout": 60}},
        }
    }
    requests = []
    response_ids = tokenizer.encode("<answer>invalid answer</answer>", add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    entered = asyncio.Event()
    release = asyncio.Event()
    hold_next = False

    async def generate(request):
        nonlocal hold_next
        body = await request.json()
        requests.append(body)
        if hold_next:
            hold_next = False
            entered.set()
            await release.wait()
        return web.json_response(
            {
                "request_id": f"fixed-{len(requests)}",
                "choices": [
                    {
                        "token_ids": response_ids,
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"logprob": -0.4 - 0.001 * i} for i in range(len(response_ids))]},
                    }
                ],
            }
        )

    async def models(request):
        return web.json_response({"data": [{"id": "rg-parity-model", "max_model_len": 32768}]})

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
        api_key_var="RG_ACCEPTANCE_NO_SECRET",
        renderer_model_name=str(args.tokenizer.resolve()),
        renderer={"name": "qwen3"},
    )
    sampling = vf.SamplingConfig(
        temperature=1,
        top_p=1,
        max_tokens=256,
        extra_body={"top_k": -1, "min_p": 0, "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0},
    )
    # Load the actual frozen ignored package, independently from native code.
    spec = importlib.util.spec_from_file_location("frozen_rg_acceptance", args.frozen_legacy)
    frozen = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = frozen
    spec.loader.exec_module(frozen)
    legacy = LegacyEnvServer(
        "frozen_rg_acceptance",
        {"dataset_path": str(args.dataset.resolve()), "num_train_examples": 10000, "num_eval_examples": 256},
        address="tcp://127.0.0.1:0",
    )
    # Legacy bridge addresses train rows by default. Append its actual heldout
    # rows solely to expose both known splits through the same actual transport.
    from datasets import concatenate_datasets

    legacy.dataset = concatenate_datasets([legacy.env.get_dataset(), legacy.env.get_eval_dataset()])
    legacy.num_tasks = len(legacy.dataset)
    legacy_task = asyncio.create_task(legacy.run())
    legacy_client = EnvClient(address=legacy.address)
    processes, clients = [], []
    ctx = mp.get_context("spawn")
    try:
        for i in range(2):
            queue = ctx.Queue()
            proc = ctx.Process(target=worker, args=(raw, queue, str(output / f"worker-{i}.jsonl")))
            proc.start()
            processes.append(proc)
            address = await asyncio.to_thread(queue.get, True, 45)
            queue.close()
            queue.join_thread()
            cli = EnvClient(address=address)
            clients.append(cli)
            await cli.wait_for_server_startup()
        results = []
        paired_raw = []
        credit_groups = [[], [], []]
        for i, task in enumerate(chosen):
            payload = json.loads(task.data.model_dump_json())
            assert "scorer_entry" in payload and getattr(task.data, "value_function_prompt", None) is None
            start = len(requests)
            old_episode = await legacy_client.run(
                task_idx=int(task.data.answer), client=client_config, model="rg-parity-model", sampling=sampling
            )
            old = project(old_episode, int(task.data.answer))
            # Same selected payload is reconstructed independently by both workers.
            native_episodes = await asyncio.gather(
                *[
                    c.run(task_data=payload, client=client_config, model="rg-parity-model", sampling=sampling)
                    for c in clients
                ]
            )
            assert all(e.traces[0].info["variant_id"] == task.data.variant_id for e in native_episodes)
            if i == 0:
                for group, episode in zip(credit_groups, [old_episode, *native_episodes], strict=True):
                    group.append(episode)
            projections = [project(e, task.data.idx) for e in native_episodes]
            assert old == projections[0] == projections[1], (
                "materialization mismatch",
                task.data.answer,
                old,
                projections,
            )
            bodies = requests[start:]
            assert len(bodies) == 3
            assert bodies[0] == bodies[1] == bodies[2], ("request mismatch", task.data.answer, bodies)
            paired_raw.append(
                {"answer_index": task.data.answer, "requests": bodies, "projections": [old, *projections]}
            )
            prompt = tokenizer.decode(bodies[0]["token_ids"], skip_special_tokens=False)
            assert task.data.prompt in prompt and task.data.system_prompt in prompt
            assert "scorer_entry" not in prompt and "variant_id" not in prompt
            assert bodies[0]["sampling_params"]["temperature"] == 1 and bodies[0]["sampling_params"]["top_p"] == 1
            assert bodies[0]["sampling_params"]["top_k"] == -1 and bodies[0]["sampling_params"]["min_p"] == 0
            assert bodies[0]["sampling_params"]["repetition_penalty"] == 1
            assert (
                bodies[0]["sampling_params"]["presence_penalty"]
                == bodies[0]["sampling_params"]["frequency_penalty"]
                == 0
            )
            results.append(
                {
                    "answer_index": task.data.answer,
                    "split_local_idx": task.data.idx,
                    "variant": task.data.variant_id,
                    "payload_sha256": digest(payload),
                    "request_sha256": digest(bodies[0]),
                    "projection_sha256": digest(old),
                    "reward": old["reward"],
                    "tokens": len(old["samples"][0]["token_ids"]),
                    "sampled_tokens": sum(old["samples"][0]["mask"]),
                }
            )
        # A second actual completion of the same cached task supplies a positive
        # peer, so the credit/LOO comparison cannot pass with all-zero rewards.
        response_ids = tokenizer.encode(
            "<answer>" + str(chosen[0].data.scorer_entry["answer"]) + "</answer>", add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        correct_legacy = await legacy_client.run(
            task_idx=int(chosen[0].data.answer), client=client_config, model="rg-parity-model", sampling=sampling
        )
        correct_native = await asyncio.gather(
            *[
                c.run(
                    task_data=chosen[0].data.model_dump(mode="json"),
                    client=client_config,
                    model="rg-parity-model",
                    sampling=sampling,
                )
                for c in clients
            ]
        )
        assert (
            project(correct_legacy, int(chosen[0].data.answer))
            == project(correct_native[0], chosen[0].data.idx)
            == project(correct_native[1], chosen[0].data.idx)
        )
        assert requests[-3] == requests[-2] == requests[-1]
        for group, episode in zip(credit_groups, [correct_legacy, *correct_native], strict=True):
            group.append(episode)
        (output / "paired-requests-projections.json").write_text(json.dumps(paired_raw, indent=2) + "\n")
        (output / "credit-episodes.json").write_text(
            json.dumps([[e.model_dump(mode="json") for e in group] for group in credit_groups], indent=2) + "\n"
        )
        credit = await credit_and_peer_context(credit_groups, tokenizer)
        (output / "credit-and-peer-context.json").write_text(json.dumps(credit, indent=2) + "\n")
        response_ids = tokenizer.encode("<answer>invalid answer</answer>", add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        # Cancellation must not poison the worker or leak a completed RG result.
        hold_next = True
        pending = asyncio.create_task(
            clients[0].run(
                task_data=chosen[0].data.model_dump(mode="json"),
                client=client_config,
                model="rg-parity-model",
                sampling=sampling,
            )
        )
        await asyncio.wait_for(entered.wait(), 30)
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending
        assert pending.cancelled()
        release.set()
        recovered = await clients[0].run(
            task_data=chosen[1].data.model_dump(mode="json"),
            client=client_config,
            model="rg-parity-model",
            sampling=sampling,
        )
        assert recovered.ok and project(recovered, chosen[1].data.idx)["reward"] == results[1]["reward"]
        assert not missing.exists()
        audits = [list(map(json.loads, (output / f"worker-{i}.jsonl").read_text().splitlines())) for i in range(2)]
        assert len({rows[0]["pid"] for rows in audits}) == 2
        for rows in audits:
            assert [r["payload_sha256"] for r in rows[: len(chosen)]] == [r["payload_sha256"] for r in results]
        report = {
            "status": "passed",
            "mixed_reward_credit_and_loo_exact": True,
            "dataset_file_sha256": source_dataset_hashes,
            "credit_and_peer_context_sha256": digest(credit),
            "provenance": provenance,
            "cases": results,
            "two_distinct_worker_pids": [rows[0]["pid"] for rows in audits],
            "worker_dataset_path_absent": True,
            "cancellation_and_recovery": True,
            "cancellation_scope": "EnvClient cancellation then worker recovery; dispatcher arrival accounting is covered separately",
            "http_requests": len(requests),
            "frozen_legacy_sha256": hashlib.sha256(args.frozen_legacy.read_bytes()).hexdigest(),
            "tokenizer_files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.tokenizer.iterdir())
                if p.is_file()
            },
            "scope": "Actual frozen legacy and native training renderers, two real worker processes with ZMQ, deterministic token endpoint, real RG scoring, exact production materialization and value snapshots; no model inference or learning-curve claim.",
        }
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "cases": len(results), "http_requests": len(requests)}))
    finally:
        release.set()
        for c in clients:
            await c.close()
        await legacy_client.close()
        legacy_task.cancel()
        await legacy_task
        for proc in processes:
            proc.terminate()
            await asyncio.to_thread(proc.join, 10)
            assert not proc.is_alive()
        await http.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--frozen-legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(asyncio.wait_for(main(parser.parse_args()), timeout=300))
