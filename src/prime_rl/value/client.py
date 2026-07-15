from __future__ import annotations

import asyncio
import time
from itertools import cycle

import httpx
import msgspec

from prime_rl.configs.value import ValueEvaluatorConfig
from prime_rl.value.types import ValueEvaluationRequest, ValueEvaluationResponse, ValueVersionResponse

_RESPONSE_GRACE_SECONDS = 5.0


class ValueEvaluatorClient:
    def __init__(self, config: ValueEvaluatorConfig):
        self.config = config
        self._urls = cycle(url.rstrip("/") for url in config.base_url)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.request_timeout + _RESPONSE_GRACE_SECONDS,
                connect=config.request_timeout,
                pool=config.request_timeout,
                write=config.request_timeout,
            )
        )
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._request_encoder = msgspec.msgpack.Encoder()
        self._evaluation_decoder = msgspec.msgpack.Decoder(type=ValueEvaluationResponse)
        self._version_decoder = msgspec.msgpack.Decoder(type=ValueVersionResponse)
        self._requests = 0
        self._sequences = 0
        self._tokens = 0
        self._errors = 0
        self._latency_seconds = 0.0
        self._max_latency_seconds = 0.0
        self._latest_version = -1

    async def wait_for_ready(self) -> None:
        async def wait(url: str) -> None:
            while True:
                try:
                    response = await self._client.get(f"{url}/health")
                    response.raise_for_status()
                    return
                except (httpx.HTTPError, OSError):
                    await asyncio.sleep(1.0)

        await asyncio.gather(*(wait(url.rstrip("/")) for url in self.config.base_url))

    async def evaluate(self, token_ids: list[list[int]]) -> ValueEvaluationResponse:
        started_at = time.perf_counter()
        self._requests += 1
        self._sequences += len(token_ids)
        self._tokens += sum(map(len, token_ids))
        try:
            async with self._semaphore:
                url = next(self._urls)
                payload = self._request_encoder.encode(ValueEvaluationRequest(token_ids=token_ids))
                response = await self._client.post(
                    f"{url}/evaluate",
                    content=payload,
                    headers={"content-type": "application/msgpack"},
                )
                response.raise_for_status()
                result = self._evaluation_decoder.decode(response.content)
            if len(result.values) != len(token_ids):
                raise ValueError(f"value evaluator returned {len(result.values)} sequences for {len(token_ids)} inputs")
            for index, (values, tokens) in enumerate(zip(result.values, token_ids, strict=True)):
                if len(values) != len(tokens):
                    raise ValueError(
                        f"value evaluator sequence {index} length {len(values)} does not match token length {len(tokens)}"
                    )
            self._latest_version = max(self._latest_version, result.version)
            return result
        except Exception:
            self._errors += 1
            raise
        finally:
            latency = time.perf_counter() - started_at
            self._latency_seconds += latency
            self._max_latency_seconds = max(self._max_latency_seconds, latency)

    async def version(self) -> int:
        versions = []
        for url in self.config.base_url:
            response = await self._client.get(f"{url.rstrip('/')}/version")
            response.raise_for_status()
            versions.append(self._version_decoder.decode(response.content).version)
        # Replicas finish one collective update a few milliseconds apart. The
        # minimum is the coherent readiness watermark: reaching N means every
        # endpoint has adopted at least N, without treating transient skew as
        # a fatal condition.
        watermark = min(versions)
        self._latest_version = max(self._latest_version, watermark)
        return watermark

    def metrics(self) -> dict[str, float]:
        """Cumulative evaluator-service metrics for orchestrator/W&B logging."""
        return {
            "value/evaluator_requests": float(self._requests),
            "value/evaluator_sequences": float(self._sequences),
            "value/evaluator_tokens": float(self._tokens),
            "value/evaluator_errors": float(self._errors),
            "value/evaluator_error_rate": self._errors / self._requests if self._requests else 0.0,
            "value/evaluator_latency_seconds_mean": (self._latency_seconds / self._requests if self._requests else 0.0),
            "value/evaluator_latency_seconds_max": self._max_latency_seconds,
            "value/evaluator_version_watermark": float(self._latest_version),
        }

    async def wait_for_version(self, minimum: int) -> None:
        while await self.version() < minimum:
            await asyncio.sleep(0.5)

    async def close(self) -> None:
        await self._client.aclose()
