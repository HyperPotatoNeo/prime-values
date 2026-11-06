"""Exercise real native and upstream legacy producers against a local deterministic HTTP endpoint.

This validates transport/config/scoring, not model tokenization or learning. The GPU
pipeline separately validates generation tokens, policy/value updates and restart.
Run from the repository with `PYTHONPATH=src uv run python experiments/episode-boundary/producer_smoke.py --output result.json`.
"""

import argparse
import asyncio
import json
from pathlib import Path

import verifiers.v1 as vf
from aiohttp import web
from pydantic import ValidationError
from verifiers.v1.legacy import LegacyEnvServer
from verifiers.v1.serve import EnvServer

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.orchestrator.envs import Env


async def check(output: Path):
    requests = []

    async def completion(request):
        payload = await request.json()
        requests.append(payload)
        return web.json_response(
            {
                "id": f"smoke-{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "seed"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = vf.EvalClientConfig(base_url=f"http://127.0.0.1:{port}/v1", api_key_var="PRODUCER_SMOKE_NO_SECRET")
    results = {}
    try:
        for name, raw in {
            "native": {
                "env": {
                    "taskset": {"id": "producer-fixture"},
                    "agent": {"harness": {"id": "null", "runtime": {"type": "subprocess"}}, "timeout": {"rollout": 90}},
                }
            },
            "legacy": {"id": "producer-fixture"},
            "legacy_group": {
                "id": "producer-fixture",
                "args": {"group_scoring": True},
                "env": {"agent": {"timeout": {"rollout": 90}, "max_output_tokens": 64}},
            },
        }.items():
            config = EnvConfig.model_validate(raw)
            assert EnvConfig.model_validate_json(config.model_dump_json()) == config
            if config.is_legacy:
                server = LegacyEnvServer(
                    config.env_id, config.args, address="tcp://127.0.0.1:0", extra_env_kwargs=config.extra_env_kwargs
                )
                assert server.dataset[0]["answer"] == "seed"
                if name == "legacy_group":
                    assert config.extra_env_kwargs == {"timeout_seconds": 90, "max_total_completion_tokens": 64}
            else:
                server = EnvServer(config.env, address="tcp://127.0.0.1:0")
                task = next(iter(vf.load_taskset(config.env.taskset).load()))
                assert (
                    server._build_task(task.data.model_dump(mode="json")).data.value_function_prompt
                    == "privileged seed"
                )
            assert server.num_tasks == (1 if config.is_legacy else None)
            server_task = asyncio.create_task(server.run())
            wrapper = Env(config.model_copy(update={"address": server.address}))
            wrapper.sampling_args = {"temperature": 1, "max_completion_tokens": 64}
            try:
                await wrapper.start(output.parent / "logs")
                assert wrapper.requires_group_scoring == (name == "legacy_group")
                if name == "legacy_group":
                    rollouts = await wrapper.run_group(client, 0, "deterministic-producer-probe", 2, None)
                    assert [rollout.reward for rollout in rollouts] == [2.0, 3.0]
                    assert all(rollout.ok and rollout.task.data.idx == 0 for rollout in rollouts)
                    results[name] = {
                        "joint_rewards": [rollout.reward for rollout in rollouts],
                        "members": len(rollouts),
                    }
                else:
                    task_data = None if config.is_legacy else next(wrapper.tasks).data.model_dump(mode="json")
                    episode = await wrapper.run_episode(
                        client, 0, "deterministic-producer-probe", None, task_data=task_data
                    )
                    assert isinstance(episode, vf.Episode) and episode.ok and len(episode.traces) == 1
                    trace = episode.traces[0]
                    assert trace.ok and trace.reward == 1 and trace.task.data.idx == 0
                    if name == "native":
                        assert trace.task.data.value_function_prompt == "privileged seed"
                    results[name] = {
                        "episode_id": episode.id,
                        "reward": trace.reward,
                        "task_type": trace.task.type,
                        "turns": trace.num_turns,
                    }
            finally:
                if wrapper._env_client is not None:
                    await wrapper.env_client.close()
                server_task.cancel()
                await server_task
        spawned = Env(
            EnvConfig.model_validate(
                {
                    "name": "native-spawn",
                    "pool": {"type": "static", "num_workers": 1},
                    "env": {
                        "taskset": {"id": "producer-fixture"},
                        "agent": {
                            "harness": {"id": "null", "runtime": {"type": "subprocess"}},
                            "timeout": {"rollout": 90},
                        },
                    },
                }
            )
        )
        spawned.sampling_args = {"temperature": 1, "max_completion_tokens": 64}
        try:
            await spawned.start(output.parent / "logs")
            task_data = next(spawned.tasks).data.model_dump(mode="json")
            episode = await spawned.run_episode(client, 0, "deterministic-producer-probe", None, task_data=task_data)
            assert episode.ok and len(episode.traces) == 1 and episode.traces[0].reward == 1
            results["native_spawn"] = {"reward": episode.traces[0].reward, "canonical_config_data": True}
        finally:
            if spawned._env_client is not None:
                await spawned.env_client.close()
            process = spawned._env_server_process
            spawned.shutdown()
            if process is not None:
                await asyncio.to_thread(process.join, 10)
                assert not process.is_alive(), "Spawned environment did not stop"
        try:
            EnvConfig.model_validate(
                {"taskset": {"id": "producer-fixture"}, "env": {"taskset": {"id": "producer-fixture"}}}
            )
        except ValidationError:
            results["ambiguous_config_rejected"] = True
        else:
            raise AssertionError("Mixed flat/canonical environment config was accepted")
        assert len(requests) == 5
        assert all(request.get("temperature") == 1 for request in requests)
        results["http_requests"] = len(requests)
        results["sampling_temperature"] = 1
        results["scope"] = (
            "Real env servers, ZMQ episodes, subprocess native harness and legacy joint scoring; deterministic HTTP model fixture, no training-token claim"
        )
        output.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(asyncio.wait_for(check(args.output), timeout=180))
