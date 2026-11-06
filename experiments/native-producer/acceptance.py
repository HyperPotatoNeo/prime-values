"""Actual old/new producer cohorts over deterministic HTTP, without learned inference."""

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import msgspec
import verifiers.v1 as vf
from aiohttp import web
from renderers.qwen3 import Qwen3Renderer
from rg_mix_v1.taskset import RGMixConfig, RGMixTaskset
from transformers import AutoTokenizer
from verifiers.v1.serve import EnvClient, EnvServer

from prime_rl.configs.orchestrator import EnvConfig, OrchestratorConfig
from prime_rl.orchestrator.envs import Env, TrainEnvs
from prime_rl.orchestrator.records import public_rollout_record
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.trajectories import materialize_training_data
from prime_rl.value.trajectory import snapshot_value_trajectory


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def worker(config, queue, audit):
    class AuditedServer(EnvServer):
        def _build_task(self, payload):
            task = super()._build_task(payload)
            with open(audit, "a") as f:
                f.write(
                    json.dumps(
                        {"pid": os.getpid(), "received": payload, "reconstructed": task.data.model_dump(mode="python")},
                        default=str,
                    )
                    + "\n"
                )
            return task

    server = AuditedServer(EnvConfig.model_validate(config).env, address="tcp://127.0.0.1:0")
    queue.put(server.address)
    asyncio.run(server.run())


def env_config(kind, stack, dataset, *, worker_side=False):
    if kind == "rg":
        taskset = {"id": "rg-mix-v1", "dataset_path": str(dataset) if not worker_side else "missing-worker-dataset"}
    else:
        taskset = {
            "id": "reverse-text-v1" if stack == "baseline" else "reverse-text",
            "dataset_name": "missing-worker-dataset",
        }
    agent = {"harness": {"id": "null"}}
    if stack == "baseline":
        agent["harness"]["runtime"] = {"type": "subprocess"}
    else:
        agent["runtime"] = {"type": "subprocess"}
    return {"name": kind, "env": {"taskset": taskset, "agent": agent}}


def training_config(kind, stack, dataset, limit, context="task", baseline="value"):
    env = env_config(kind, stack, dataset)
    baseline_config = {"type": baseline}
    if baseline == "tether":
        baseline_config["adaptive"] = {"batch_size": 2}
    env.update(
        algo={"type": "grpo", "baseline": baseline_config, "branch_semantics": "sequential"},
        training_projection={"type": "single_turn"},
    )
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
    assert config.train.env[0].training_projection.max_seq_len == limit
    return config


class RecordedEnv(Env):
    async def run_episode(self, *args, **kwargs):
        self.episode = await super().run_episode(*args, **kwargs)
        self.before = self.episode.model_dump(mode="python")
        return self.episode


class RecordedClient(EnvClient):
    def __init__(self, address, stack):
        super().__init__(address=address)
        self.stack = stack
        self.deltas = []
        self.delta_event = asyncio.Event()

    async def run(self, *args, **kwargs):
        if self.stack == "candidate":

            def record(delta):
                self.deltas.append(delta)
                self.delta_event.set()

            kwargs["on_delta"] = record
        return await super().run(*args, **kwargs)


def projection(rollout):
    materialized = materialize_training_data(rollout, env_name="cohort", capture_sampled_masks=True)
    owned = snapshot_value_trajectory(
        materialized.samples,
        sampled_masks=materialized.sampled_masks,
        layout=materialized.layout,
        reward=rollout.reward,
        env_name="cohort",
        group_id="fixed",
        policy_version=3,
    )
    result = {
        "reward": rollout.reward,
        "samples": [
            {"token_ids": list(s.token_ids), "mask": list(s.mask), "logprobs": list(s.logprobs)}
            for s in materialized.samples
        ],
        "owned": asdict(owned),
        "is_completed": rollout.is_completed,
        "is_truncated": rollout.is_truncated,
        "num_output_tokens": rollout.num_output_tokens,
        "num_total_tokens": rollout.num_total_tokens,
        "usage": rollout.usage.model_dump(mode="python"),
        "call_finish_reasons": [c.finish_reason for c in rollout.calls],
        "metrics": {
            k: rollout.metrics[k]
            for k in ["check_answer_reward_func", "format_reward_func", "lcs"]
            if k in rollout.metrics
        },
    }
    prior = asdict(owned)
    for s in materialized.samples:
        s.token_ids.clear()
        s.mask.clear()
        s.logprobs.clear()
    assert asdict(owned) == prior
    assert result["samples"] and sum(result["samples"][0]["mask"]) > 0
    return result


async def credit(group, tokenizer, config):
    requests, prompts, published = [], [], []

    class Evaluator:
        async def evaluate(self, token_ids):
            requests.append([list(row) for row in token_ids])
            return SimpleNamespace(values=[[0.0] + [0.01 * (t % 11) for t in row[1:]] for row in token_ids], version=7)

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
    try:
        rounds = []
        for round_idx in range(2 if config.train.env[0].algo.baseline.type == "tether" else 1):
            rollouts = [r.model_copy(deep=True) for r in group]
            for r in rollouts:
                r.env_name = config.train.env[0].name
                r.group_id = uuid.UUID(int=round_idx + 1)
                r.policy_version = 3
            assert await sink.add(rollouts[0]) is None
            if id(rollouts[0]) in sink.scoring_tasks:
                await sink.scoring_tasks[id(rollouts[0])]
            batch = await sink.add(rollouts[1])
            assert batch is not None
            rounds.append(
                {
                    "rollouts": [
                        {
                            "reward": r.reward,
                            "values": r.value_predictions,
                            "advantages": r.value_advantages,
                            "targets": r.value_returns,
                        }
                        for r in rollouts
                    ],
                    "policy_samples": [msgspec.to_builtins(s) for s in batch.samples],
                    "algorithm_state": list(envs)[0].algorithm.state_dict(),
                }
            )
            if round_idx == 0 and config.train.env[0].algo.baseline.type == "tether":
                saved = json.loads(json.dumps(list(envs)[0].algorithm.state_dict()))
                replacement = TrainEnvs(
                    config.train.env, policy_pool=None, value_evaluator=Evaluator(), value_config=config.value_function
                )
                restored = list(replacement)[0].algorithm
                restored.load_state_dict(saved)
                assert restored.state_dict() == saved
                list(envs)[0].algorithm = restored
        result = {
            "requests": requests,
            "prompts": prompts,
            "published": [[msgspec.to_builtins(s) for s in x.samples] for x in published],
            "rounds": rounds,
        }
        frozen = json.loads(json.dumps(result))
        for r in rollouts:
            r.nodes.clear()
            r.samples.clear()
            r.rewards.clear()
        assert result == frozen
        return result
    finally:
        await sink.stop()


def source_guard(root):
    files = {}
    for module_name in [
        "prime_rl.orchestrator.envs",
        "prime_rl.orchestrator.episodes",
        "prime_rl.orchestrator.types",
        "prime_rl.orchestrator.trajectories",
        "prime_rl.orchestrator.train_sink",
        "prime_rl.orchestrator.value_context",
        "prime_rl.orchestrator.records",
        "prime_rl.orchestrator.algo.grpo",
        "prime_rl.orchestrator.algo.tether",
        "prime_rl.value.credit",
        "prime_rl.value.trajectory",
        "prime_rl.configs.orchestrator",
    ]:
        module = importlib.import_module(module_name)
        p = Path(module.__file__).resolve()
        assert p.is_relative_to(root), (module_name, str(p), str(root))
        files[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    dependency_code = {}
    for name in ["verifiers", "renderers", "pydantic_config"]:
        directory = Path(importlib.import_module(name).__file__).parent
        dependency_code[name] = {
            str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*.py")
        }
    for name, relative in [
        ("rg_mix_v1.taskset", "rg_mix_v1/taskset.py"),
        ("rg_mix_v1.scoring", "rg_mix_v1/scoring.py"),
        ("rg_mix_shared", "rg_mix_shared.py"),
    ]:
        actual = Path(importlib.import_module(name).__file__)
        expected = root / "packages/rg-mix-env" / relative
        assert actual.read_bytes() == expected.read_bytes(), name
        files[str(expected.relative_to(root))] = hashlib.sha256(actual.read_bytes()).hexdigest()
    for relative in [
        "src/prime_rl/orchestrator/env_server/env_server.py",
        "src/prime_rl/orchestrator/dispatcher.py",
        "src/prime_rl/utils/monitor/prime.py",
    ]:
        files[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    return {
        "dependency_code": dependency_code,
        "source": str(root),
        "files": files,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in [
                "verifiers",
                "renderers",
                "prime-pydantic-config",
                "prime",
                "torch",
                "transformers",
                "numpy",
                "reasoning-gym",
            ]
        },
    }


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = source_guard(args.source)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    rg = RGMixTaskset(RGMixConfig(dataset_path=args.dataset, num_train_examples=1)).load()[0].data
    reverse_module = importlib.import_module(
        "reverse_text_v1.taskset" if args.stack == "baseline" else "reverse_text.taskset"
    )
    reverse = reverse_module.ReverseTextData(
        idx=17,
        prompt="Value functions preserve order.",
        system_prompt=reverse_module.SYSTEM,
        answer="Value functions preserve order."[::-1],
    )
    fixtures = {
        "rg": (rg, "<answer>" + str(rg.scorer_entry["answer"]) + "</answer>", "<answer>incorrect</answer>"),
        "reverse": (
            reverse,
            "<reversed_text>" + reverse.answer + "</reversed_text>",
            "<reversed_text></reversed_text>",
        ),
    }
    requests = []
    state = {"tokens": [], "mode": "normal"}
    entered = asyncio.Event()
    release = asyncio.Event()

    async def generate(request):
        requests.append(await request.json())
        entered.set()
        if state["mode"] == "hold":
            await release.wait()
        if state["mode"] == "failure":
            return web.json_response({"request_id": "fixed", "choices": []})
        return web.json_response(
            {
                "request_id": "fixed",
                "choices": [
                    {
                        "token_ids": state["tokens"],
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {
                                    "token": f"token_id:{state['tokens'][i]}",
                                    "logprob": -0.123456789123 - i * 0.000000001,
                                }
                                for i in range(len(state["tokens"]))
                            ]
                        },
                    }
                ],
            }
        )

    async def models(request):
        return web.json_response({"data": [{"id": "cohort-model", "max_model_len": 32768}]})

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
        api_key_var="COHORT_NO_SECRET",
        renderer_model_name=str(args.tokenizer),
        renderer={"name": "qwen3"},
    )
    sampling = {
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 256,
        "extra_body": {"top_k": -1, "min_p": 0, "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0},
    }
    all_cases = []
    control = {}
    child_context = mp.get_context("spawn")
    try:
        for kind, (data, correct, incorrect) in fixtures.items():
            processes = []
            clients = []
            try:
                for worker_idx in range(2):
                    queue = child_context.Queue()
                    p = child_context.Process(
                        target=worker,
                        args=(
                            env_config(kind, args.stack, args.dataset, worker_side=True),
                            queue,
                            args.output / f"{kind}-worker{worker_idx}.jsonl",
                        ),
                    )
                    p.start()
                    processes.append(p)
                    address = await asyncio.wait_for(asyncio.to_thread(queue.get, True, 40), 45)
                    c = RecordedClient(address, args.stack)
                    clients.append(c)
                    await c.wait_for_server_startup(timeout=20)
                payload = data.model_dump(mode="python")
                payload_digest = digest(payload)
                response_ids = [tokenizer.encode(x, add_special_tokens=False) for x in (correct, incorrect)]
                width = max(map(len, response_ids)) + 2
                tail = tokenizer.encode(" trailing", add_special_tokens=False)
                assert len(tail) == 1
                response_ids = [x + tail * (width - len(x)) + [tokenizer.eos_token_id] for x in response_ids]
                physical = []
                full_group = []

                async def produce(client, limit, tokens, task_payload=payload):
                    state["tokens"] = tokens
                    config = training_config(kind, args.stack, args.dataset, limit)
                    env = RecordedEnv(config.train.env[0])
                    env._env_client = client
                    env.sampling_args = sampling
                    rollout = await asyncio.wait_for(
                        env.run_rollout(client_config, data.idx, "cohort-model", None, task_data=task_payload), 40
                    )
                    (args.output / f"episode-{kind}-{len(requests)}.json").write_text(
                        json.dumps({"episode": env.before, "requests": requests}, indent=2, default=str) + "\n"
                    )
                    assert env.episode.model_dump(mode="python") == env.before
                    return rollout, env.before

                for member, tokens in enumerate(response_ids):
                    start = len(requests)
                    r, raw = await produce(clients[member], 32768, tokens)
                    assert r.ok
                    full_group.append(r)
                    physical.append(raw)
                    if member == 0:
                        prompt_len = len(requests[start]["token_ids"])
                assert [r.reward for r in full_group] == [1, 0], kind
                for case, limit in [("within", 32768), ("clipped", prompt_len + 3)]:
                    group = full_group if case == "within" else []
                    raw_cases = physical if case == "within" else []
                    if case == "clipped":
                        for member, tokens in enumerate(response_ids):
                            r, raw = await produce(clients[member], limit, tokens)
                            group.append(r)
                            raw_cases.append(raw)
                    records = [projection(r.model_copy(deep=True)) for r in group]
                    assert [sum(s["mask"]) for s in records[0]["samples"]] == [
                        len(response_ids[0]) if case == "within" else 3
                    ]
                    contexts = {}
                    for context, baseline in [("task", "value"), ("group_leave_one_out", "value"), ("task", "tether")]:
                        contexts[context + "/" + baseline] = await credit(
                            group, tokenizer, training_config(kind, args.stack, args.dataset, limit, context, baseline)
                        )
                    all_cases.append(
                        {
                            "env": kind,
                            "case": case,
                            "payload": payload,
                            "payload_sha256": payload_digest,
                            "records": records,
                            "physical_episodes": raw_cases,
                            "credit": contexts,
                        }
                    )
                    (args.output / "partial.json").write_text(
                        json.dumps({"cases": all_cases, "requests": requests}, indent=2, default=str) + "\n"
                    )
                if args.stack == "candidate" and kind == "rg":
                    private = "CRITIC_ONLY_PREFIX_SENTINEL: 17 plus 25 equals 42."
                    private_payload = {**payload, "value_function_prompt": private}
                    private_group = []
                    private_episodes = []
                    public = []
                    private_start = len(requests)
                    for member, tokens in enumerate(response_ids):
                        r, raw = await produce(clients[member], 32768, tokens, private_payload)
                        assert r.task.data.value_function_prompt == private
                        assert raw["traces"][0]["task"]["data"]["value_function_prompt"] == private
                        redacted = public_rollout_record(r)
                        assert private not in json.dumps(redacted)
                        private_group.append(r)
                        private_episodes.append(raw)
                        public.append(redacted)
                    assert [x["token_ids"] for x in requests[private_start:]] == [x["token_ids"] for x in requests[:2]]
                    private_credit = await credit(
                        private_group, tokenizer, training_config(kind, args.stack, args.dataset, 32768)
                    )
                    assert private_credit["prompts"] == [private, private]
                    prefix_ids = Qwen3Renderer(tokenizer).render_ids(
                        [{"role": "system", "content": private}], add_generation_prompt=False
                    )
                    assert tokenizer.bos_token_id is None
                    for member in range(2):
                        sample = materialize_training_data(full_group[member], env_name="rg").samples[0]
                        assert private_credit["requests"][member] == [prefix_ids + sample.token_ids]
                    control["private_prefix"] = {
                        "episodes": private_episodes,
                        "public": public,
                        "credit": private_credit,
                        "requests": requests[private_start:],
                    }
                    state["mode"] = "failure"
                    failed, failed_raw = await produce(clients[0], 32768, response_ids[0])
                    assert not failed.ok and not materialize_training_data(failed, env_name="rg").samples
                    control["failed_episode"] = {
                        "raw": failed_raw,
                        "rollout_ok": failed.ok,
                        "public": public_rollout_record(failed),
                    }
                    state["mode"] = "normal"
                if args.stack == "candidate" and kind == "rg":
                    local_config = training_config(kind, args.stack, args.dataset, 32768)
                    local_config.train.env[0].env.taskset.num_train_examples = 1
                    local_config.train.env[0].pool = vf.StaticPoolConfig(num_workers=1)
                    local = RecordedEnv(local_config.train.env[0])
                    try:
                        await asyncio.wait_for(local.start(args.output / "production-start"), 45)
                        selected = next(local.tasks)
                        assert selected.data.idx == 0
                        local.sampling_args = sampling
                        state["tokens"] = response_ids[0]
                        local_rollout = await asyncio.wait_for(
                            local.run_rollout(
                                client_config,
                                selected.data.idx,
                                "cohort-model",
                                None,
                                task_data=selected.data.model_dump(mode="python"),
                            ),
                            40,
                        )
                        assert local_rollout.ok and local_rollout.reward == 1
                        control["production_start"] = {
                            "num_tasks": local.num_tasks,
                            "pid": local._env_server_process.pid,
                            "projection": projection(local_rollout.model_copy(deep=True)),
                        }
                    finally:
                        if local._env_client is not None:
                            await local._env_client.close()
                        local_process = local._env_server_process
                        if local_process is not None:
                            if local_process.is_alive():
                                local_process.terminate()
                            await asyncio.to_thread(local_process.join, 10)
                            if local_process.is_alive():
                                local_process.kill()
                                await asyncio.to_thread(local_process.join, 5)
                            assert not local_process.is_alive()
                # Cancellation is a client/server control; a subsequent successful run proves worker reuse.
                entered.clear()
                release.clear()
                clients[0].delta_event.clear()
                state["mode"] = "hold"
                pending = asyncio.create_task(produce(clients[0], 32768, response_ids[0]))
                await asyncio.wait_for(entered.wait(), 10)
                if args.stack == "candidate":
                    await asyncio.wait_for(clients[0].delta_event.wait(), 10)
                    assert not pending.done(), "A partial streamed trace escaped as a final rollout"
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending
                state["mode"] = "normal"
                release.set()
                recovered, _ = await produce(clients[0], 32768, response_ids[0])
                assert recovered.reward == 1
                control[kind] = {
                    "cancelled": True,
                    "recovery_reward": recovered.reward,
                    "worker_pids": [p.pid for p in processes],
                    "delta_count": sum(len(c.deltas) for c in clients),
                    "partial_not_final": args.stack == "candidate",
                }
                if args.stack == "candidate":
                    assert control[kind]["delta_count"] > 0
            finally:
                for c in clients:
                    await c.close()
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                for p in processes:
                    await asyncio.to_thread(p.join, 10)
                    if p.is_alive():
                        p.kill()
                        await asyncio.to_thread(p.join, 5)
                    assert not p.is_alive()
        worker_audit = {}
        for path in sorted(args.output.glob("*-worker*.jsonl")):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert rows and all(row["received"] == row["reconstructed"] for row in rows)
            worker_audit[path.name] = {"requests": len(rows), "pid": rows[0]["pid"], "exact_payload_roundtrip": True}
        assert len(worker_audit) == 4 and len({x["pid"] for x in worker_audit.values()}) == 4
        assert source_guard(args.source) == provenance, "Source or loaded dependency bytes changed during acceptance"
        raw = {"cases": all_cases, "requests": requests, "controls": control, "worker_audit": worker_audit}
        (args.output / "raw.json").write_text(json.dumps(raw, indent=2, default=str) + "\n")
        result = {
            "status": "passed",
            "stack": args.stack,
            "provenance": provenance,
            "case_count": len(all_cases),
            "http_requests": len(requests),
            "controls": {k: (v if k in ["rg", "reverse"] else {"passed": True}) for k, v in control.items()},
            "worker_audit": worker_audit,
            "raw_sha256": hashlib.sha256((args.output / "raw.json").read_bytes()).hexdigest(),
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {k: result[k] for k in ["status", "stack", "case_count", "http_requests", "controls", "raw_sha256"]}
            )
        )
    finally:
        await http.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=["baseline", "candidate"], required=True)
    for name in ["source", "dataset", "tokenizer", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.source = args.source.resolve()
    asyncio.run(main(args))
