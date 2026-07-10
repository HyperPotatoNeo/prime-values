from __future__ import annotations

import asyncio
from itertools import cycle

import httpx
import msgspec

from prime_rl.configs.value import ValueEvaluatorConfig
from prime_rl.value.types import ValueEvaluationRequest, ValueEvaluationResponse, ValueVersionResponse


class ValueEvaluatorClient:
    def __init__(self, config: ValueEvaluatorConfig):
        self.config = config
        self._urls = cycle(url.rstrip("/") for url in config.base_url)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(config.request_timeout))
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._request_encoder = msgspec.msgpack.Encoder()
        self._evaluation_decoder = msgspec.msgpack.Decoder(type=ValueEvaluationResponse)
        self._version_decoder = msgspec.msgpack.Decoder(type=ValueVersionResponse)

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
        return result

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
        return min(versions)

    async def wait_for_version(self, minimum: int) -> None:
        while await self.version() < minimum:
            await asyncio.sleep(0.5)

    async def close(self) -> None:
        await self._client.aclose()
