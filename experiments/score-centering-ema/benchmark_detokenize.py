"""Paired fixed-length requests: text bookkeeping versus token-only serving."""

import asyncio
import json
import time
from pathlib import Path

import httpx
import msgspec
import numpy as np

from prime_rl.transport.types import TrainingBatch


async def main():
    root = Path("/data/outputs/siddarth-rg-sc-s11-k128-b0-perf-smoke/run_default/rollouts")
    file = next(root.glob("step_*/train_rollouts.bin"))
    batch = msgspec.msgpack.decode(file.read_bytes(), type=TrainingBatch)
    prompts = [s.token_ids[: s.mask.index(True)] for s in batch.examples[:32]]
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=40)) as client:
        for _ in range(120):
            try:
                if (await client.get("http://localhost:8000/health")).status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            await asyncio.sleep(2)
        results = []
        payloads = []
        for detokenize, kind in [(True, 0), (False, 2)]:

            async def request(i, prompt):
                sp = {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "max_tokens": 2048,
                    "ignore_eos": True,
                    "seed": 1000 + i,
                    "logprobs": 128,
                    "detokenize": detokenize,
                    "output_kind": kind,
                    "extra_args": {"score_centering_top_k": 128},
                }
                r = await client.post(
                    "http://localhost:8000/inference/v1/generate", json={"token_ids": prompt, "sampling_params": sp}
                )
                r.raise_for_status()
                return r.json()["choices"][0]

            start = time.perf_counter()
            outputs = await asyncio.gather(*(request(i, p) for i, p in enumerate(prompts)))
            elapsed = time.perf_counter() - start
            payloads.append(outputs)
            results.append(
                {
                    "detokenize": detokenize,
                    "output_kind": kind,
                    "seconds": elapsed,
                    "tokens": sum(len(o["token_ids"]) for o in outputs),
                    "requests": len(outputs),
                }
            )
            print(results[-1], flush=True)
        compare = []
        for a, b in zip(*payloads):
            compare.append(
                {
                    "identical_tokens": a["token_ids"] == b["token_ids"],
                    "identical_head": a["sampler_head"] == b["sampler_head"],
                    "max_chosen_logprob_delta": float(
                        np.max(
                            np.abs(
                                np.array([x["logprob"] for x in a["logprobs"]["content"]])
                                - np.array([x["logprob"] for x in b["logprobs"]["content"]])
                            )
                        )
                    ),
                }
            )
        receipt = {
            "measurements": results,
            "comparisons": compare,
            "note": "Fixed 2048-token ignore-EOS throughput probe only; production sampling unchanged.",
        }
        Path("/data/outputs/siddarth-performance-detokenize.json").write_text(json.dumps(receipt, indent=2))
        print(compare, flush=True)


asyncio.run(main())
