import asyncio
from unittest.mock import AsyncMock

import httpx
import msgspec

from prime_rl.configs.value import ValueEvaluatorConfig
from prime_rl.value.client import ValueEvaluatorClient
from prime_rl.value.types import ValueVersionResponse


def test_version_uses_minimum_replica_watermark_during_update_skew():
    async def run_test() -> None:
        client = ValueEvaluatorClient(ValueEvaluatorConfig(base_url=["http://eval-0:1", "http://eval-1:2"]))
        encoder = msgspec.msgpack.Encoder()
        client._client.get = AsyncMock(
            side_effect=[
                httpx.Response(
                    200,
                    content=encoder.encode(ValueVersionResponse(version=4)),
                    request=httpx.Request("GET", "http://eval-0:1/version"),
                ),
                httpx.Response(
                    200,
                    content=encoder.encode(ValueVersionResponse(version=5)),
                    request=httpx.Request("GET", "http://eval-1:2/version"),
                ),
            ]
        )

        assert await client.version() == 4
        await client.close()

    asyncio.run(run_test())
