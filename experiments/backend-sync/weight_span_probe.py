"""Real full-weight reloads during one natural response, then canonical token attribution.

Run against a fresh, otherwise idle PrimeRL filesystem-update server. No synthetic
token response, minimum-output constraint, sampling-mask replay or label-only update.
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx
import msgspec
import numpy as np
import verifiers.v1 as vf
from openai import AsyncOpenAI
from verifiers.v1.clients.train import TrainClient
from verifiers.v1.configs.agent import WireAgentConfig
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.graph import prepare_turn
from verifiers.v1.serve.delta import dump

import prime_rl.orchestrator.episodes as episodes_module
import prime_rl.orchestrator.provenance as provenance_module
import prime_rl.orchestrator.trajectories as trajectories_module
import prime_rl.utils.client as client_module
from prime_rl.orchestrator.episodes import from_episode
from prime_rl.orchestrator.trajectories import materialize_training_data
from prime_rl.utils.client import update_weights

LONG_PROMPT = (
    "Write a detailed, self-contained field guide to a fictional coastal city. "
    "Cover its geography, twelve distinct neighborhoods, local history, public transport, "
    "architecture, markets, parks, museums, seasonal festivals, practical travel advice, "
    "and several day-by-day walking itineraries. Give concrete names and sensory descriptions. "
    "Aim for about 2500 words of continuous useful prose organized into clear sections."
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


async def main(args):
    assert not args.output.exists(), "Use a fresh evidence directory"
    args.output.mkdir(parents=True)
    root = args.source_root.resolve()
    modules = (episodes_module, provenance_module, trajectories_module, client_module)
    sources = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(root), path
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for checkpoint in (args.checkpoint3, args.checkpoint4):
        assert checkpoint.is_dir() and list(checkpoint.glob("*.safetensors")), checkpoint
    events = []
    requests = []
    responses = []

    async def record_request(request):
        if request.url.path.endswith("/inference/v1/generate"):
            requests.append(json.loads(request.content))

    async def record_response(response):
        if response.request.url.path.endswith("/inference/v1/generate"):
            await response.aread()
            responses.append(response.json())

    config = vf.TrainClientConfig(
        base_url=args.base_url.rstrip("/") + "/v1",
        api_key_var="SPAN_PROBE_UNUSED_KEY",
        renderer_model_name=args.tokenizer,
        renderer={"name": "qwen3"},
    )
    client = TrainClient(config)
    await client.client.close()
    client.client = AsyncOpenAI(
        base_url=config.base_url,
        api_key="unused-local-server",
        max_retries=0,
        http_client=httpx.AsyncClient(
            timeout=600,
            event_hooks={"request": [record_request], "response": [record_response]},
        ),
    )
    async with httpx.AsyncClient(base_url=args.base_url, timeout=800) as admin:

        async def get(path):
            response = await admin.get(path)
            response.raise_for_status()
            return response.json()

        async def post(path, **kwargs):
            response = await admin.post(path, **kwargs)
            response.raise_for_status()
            return response

        async def metrics():
            response = await admin.get("/metrics")
            response.raise_for_status()
            values = [
                float(line.split()[-1])
                for line in response.text.splitlines()
                if line.startswith("vllm:generation_tokens_total{") or line.startswith("vllm:generation_tokens_total ")
            ]
            assert values, "Actual generation counter must be exposed"
            return sum(values)

        async def generate(name, prompt, max_tokens, expected_versions):
            start = len(responses)
            trace = vf.WireTrace(
                task=vf.TraceTask(type="NaturalTextProbe", data=vf.WireTaskData(idx=0, prompt=prompt)),
                agent=vf.AgentInfo(config=WireAgentConfig(model=args.model, runtime={"type": "subprocess"})),
            )
            messages = [vf.UserMessage(content=prompt)]
            turn = prepare_turn(trace, messages, dialect="chat_completions")
            sampling = vf.SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                min_p=0.0,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                max_tokens=max_tokens,
            )
            response = await client.get_response(
                ChatDialect(),
                {"model": args.model, "messages": [{"role": "user", "content": prompt}]},
                sampling,
                turn=turn,
            )
            node_id = turn.commit(response)
            trace.calls.append(
                vf.ModelCall(
                    node=node_id,
                    model=args.model,
                    sampling=sampling,
                    usage=response.usage,
                    finish_reason=response.finish_reason,
                )
            )
            trace.ok = trace.is_completed = True
            episode = vf.WireEpisode(task=trace.task, traces=[trace], ok=True)
            assert len(responses) == start + 1
            raw = responses[-1]
            request = requests[-1]
            save(args.output / f"{name}-request.json", request)
            save(args.output / f"{name}-response.json", raw)
            choice = raw["choices"][0]
            spans = choice["weight_versions"]
            n = len(choice["token_ids"])
            assert n > 0 and spans
            expected = [-1] * n
            end = 0
            for span in spans:
                assert span["start"] == end and span["end"] > end
                end = span["end"]
                assert end <= n
                version = span["version"]
                expected[span["start"] : end] = [int(version) if version in ("3", "4") else -1] * (end - span["start"])
            assert end == n
            assert {span["version"] for span in spans} == set(expected_versions)
            rollout = from_episode(episode)
            materialized = materialize_training_data(rollout, capture_policy_versions=True)
            assert len(materialized.samples) == 1
            sample = materialized.samples[0]
            assert sample.token_ids == request["token_ids"] + choice["token_ids"]
            wanted = [-1] * len(request["token_ids"]) + expected
            payload = sample.token_policy_versions
            actual = [-1] * len(wanted) if payload is None else np.frombuffer(payload.data, dtype=np.int64).tolist()
            assert actual == wanted
            raw_lps = [entry["logprob"] for entry in choice["logprobs"]["content"]]
            assert [lp for lp, mask in zip(sample.logprobs, sample.mask) if mask] == raw_lps
            assert request["sampling_params"]["temperature"] == request["sampling_params"]["top_p"] == 1.0
            assert request["sampling_params"]["top_k"] == -1
            assert request["sampling_params"]["min_p"] == 0
            assert not request["sampling_params"].get("min_tokens")
            (args.output / f"{name}-episode.msgpack").write_bytes(msgspec.msgpack.encode(dump(episode)))
            save(args.output / f"{name}-carrier.json", actual)
            events.append({"case": name, "completion_tokens": n, "spans": spans})
            return raw

        pending = None
        try:
            assert (await get("/weight_info"))["weight_version"] == "default"
            await generate("startup", "Describe a lighthouse in three sentences.", 128, ["default"])
            await update_weights([admin], args.checkpoint3, step=3)
            assert (await get("/weight_info"))["weight_version"] == "3"
            await generate("adopted3", "Describe a harbor in three sentences.", 128, ["3"])
            # Drain preceding completed calls from the metric stream so the next
            # increment establishes progress of this sole active request.
            previous_tokens = sum(event.get("completion_tokens", 0) for event in events)
            deadline = time.monotonic() + 30
            while (before := await metrics()) < previous_tokens:
                assert time.monotonic() < deadline, "Completed-call metrics did not arrive"
                await asyncio.sleep(0.05)
            assert before == previous_tokens, "Server must be otherwise idle and freshly started"
            pending = asyncio.create_task(generate("crossing", LONG_PROMPT, 4096, ["3", "4"]))
            deadline = time.monotonic() + 180
            while True:
                if pending.done():
                    await pending
                    raise AssertionError("Natural response finished before observed reload; no crossing proof")
                observed = await metrics()
                if observed > before:
                    break
                assert time.monotonic() < deadline, "No actual decode progress observed"
                await asyncio.sleep(0.05)
            events.append({"case": "decode_observed", "before": before, "after": observed})
            await update_weights([admin], args.checkpoint4, step=4)
            assert (await get("/weight_info"))["weight_version"] == "4"
            await asyncio.wait_for(pending, 600)
            pending = None
            missing = args.output / "nonexistent-checkpoint"
            try:
                await update_weights([admin], missing, step=99)
            except httpx.HTTPStatusError as error:
                save(
                    args.output / "failed-reload.json",
                    {"status": error.response.status_code, "body": error.response.text},
                )
            else:
                raise AssertionError("Missing checkpoint unexpectedly adopted")
            assert (await get("/is_paused"))["is_paused"] is True
            assert (await get("/weight_info"))["weight_version"] == "4"
            events.append({"case": "failed_filesystem_reload", "paused": True, "version": "4"})
            # Restore valid weights before resuming; the failed load may have partially mutated model state.
            await update_weights([admin], args.checkpoint4, step=4)
            await post("/pause")
            await post("/update_weights", json={"weight_dir": str(args.checkpoint4)})
            assert (await get("/weight_info"))["weight_version"] == "default"
            await post("/resume")
            await generate("unlabelled", "Describe a garden in three sentences.", 128, ["default"])
            save(
                args.output / "result.json",
                {
                    "status": "passed",
                    "source_root": str(root),
                    "sources": sources,
                    "checkpoints": [str(args.checkpoint3), str(args.checkpoint4)],
                    "events": events,
                    "scope": "Real filesystem adoption and naturally generated version crossing; real renderer, graph commit, Episode and materializer. No new value objective, environment scoring, replay or NCCL failure injection.",
                },
            )
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await client.close()
            save(args.output / "events.json", events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint3", type=Path, required=True)
    parser.add_argument("--checkpoint4", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
