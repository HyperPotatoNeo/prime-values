"""Compare identical sampler output through the original and compact response paths."""

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

from prime_rl.inference.vllm.serving_tokens import PrimeRlServingTokens

spec = importlib.util.spec_from_file_location(
    "old_serving_tokens", "/data/siddarth-sc-ema/src/prime_rl/inference/vllm/serving_tokens.py"
)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
row = {i: Logprob(logprob=-5.0 - i / 100, rank=i + 1) for i in range(128)}
row[999] = Logprob(logprob=-12.0, rank=1000)
ids = [0, 999] * 4096
output = CompletionOutput(
    index=0, text="", token_ids=ids, cumulative_logprob=None, logprobs=[row] * len(ids), finish_reason="length"
)
final = RequestOutput(
    request_id="benchmark", prompt=None, prompt_token_ids=[1], prompt_logprobs=None, outputs=[output], finished=True
)
request = GenerateRequest(
    token_ids=[1], sampling_params=SamplingParams(logprobs=128, extra_args={"score_centering_top_k": 128})
)


async def build(cls):
    server = object.__new__(cls)
    server.model_config = SimpleNamespace(enable_return_routed_experts=False)
    server.enable_prompt_tokens_details = False
    server.enable_log_outputs = False

    async def source():
        assert request.sampling_params.logprobs == 128
        yield final

    start = time.perf_counter()
    result = await server.serve_tokens_full_generator(
        request, source(), "benchmark", "model", RequestResponseMetadata(request_id="benchmark")
    )
    payload = result.model_dump(exclude={"request_id"})
    elapsed = time.perf_counter() - start
    return payload, elapsed


before, before_time = asyncio.run(build(old.PrimeRlServingTokens))
after, after_time = asyncio.run(build(PrimeRlServingTokens))
assert before == after
receipt = {
    "tokens": 8192,
    "top_k": 128,
    "old_seconds": before_time,
    "new_seconds": after_time,
    "speedup": before_time / after_time,
    "all_response_fields_equal_except_random_request_id": True,
}
Path("/data/outputs/siddarth-performance-response.json").write_text(json.dumps(receipt, indent=2))
print(receipt)
