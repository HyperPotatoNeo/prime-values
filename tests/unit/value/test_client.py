import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import msgspec
import pytest

from prime_rl.configs.value import ValueEvaluatorConfig
from prime_rl.value.client import ValueEvaluatorClient
from prime_rl.value.types import ValueEvaluationResponse, ValueVersionResponse


def test_client_adds_response_grace_only_to_the_read_timeout():
    async def run_test() -> None:
        client = ValueEvaluatorClient(ValueEvaluatorConfig(request_timeout=12.0))

        assert client._client.timeout.read == 17.0
        assert client._client.timeout.connect == 12.0
        assert client._client.timeout.pool == 12.0
        assert client._client.timeout.write == 12.0
        await client.close()

    asyncio.run(run_test())


def test_wait_for_ready_retries_server_errors():
    async def run_test() -> None:
        client = ValueEvaluatorClient(ValueEvaluatorConfig(base_url=["http://eval:1"]))
        client._client.get = AsyncMock(
            side_effect=[
                httpx.Response(503, request=httpx.Request("GET", "http://eval:1/health")),
                httpx.Response(200, request=httpx.Request("GET", "http://eval:1/health")),
            ]
        )

        with patch("prime_rl.value.client.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await client.wait_for_ready()

        sleep.assert_awaited_once_with(1.0)
        assert client._client.get.await_count == 2
        await client.close()

    asyncio.run(run_test())


def test_wait_for_ready_rejects_client_errors():
    async def run_test() -> None:
        client = ValueEvaluatorClient(ValueEvaluatorConfig(base_url=["http://eval:1"]))
        client._client.get = AsyncMock(
            return_value=httpx.Response(404, request=httpx.Request("GET", "http://eval:1/health"))
        )

        with patch("prime_rl.value.client.asyncio.sleep", new_callable=AsyncMock) as sleep:
            sleep.side_effect = AssertionError("client errors must not be retried")
            with pytest.raises(httpx.HTTPStatusError):
                await client.wait_for_ready()

        sleep.assert_not_awaited()
        assert client._client.get.await_count == 1
        await client.close()

    asyncio.run(run_test())


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
        metrics = client.metrics()
        assert metrics["value/evaluator_version_watermark"] == 4
        assert metrics["value/evaluator_response_version_latest"] == -1
        await client.close()

    asyncio.run(run_test())


def test_version_metric_tracks_the_latest_coherent_replica_poll():
    async def run_test() -> None:
        client = ValueEvaluatorClient(ValueEvaluatorConfig(base_url=["http://eval:1"]))
        encoder = msgspec.msgpack.Encoder()
        client._client.get = AsyncMock(
            side_effect=[
                httpx.Response(
                    200,
                    content=encoder.encode(ValueVersionResponse(version=4)),
                    request=httpx.Request("GET", "http://eval:1/version"),
                ),
                httpx.Response(
                    200,
                    content=encoder.encode(ValueVersionResponse(version=3)),
                    request=httpx.Request("GET", "http://eval:1/version"),
                ),
            ]
        )

        assert await client.version() == 4
        assert await client.version() == 3
        assert client.metrics()["value/evaluator_version_watermark"] == 3
        await client.close()

    asyncio.run(run_test())


def test_evaluation_metrics_cover_volume_latency_errors_and_version():
    async def run_test() -> None:
        client = ValueEvaluatorClient(ValueEvaluatorConfig(base_url=["http://eval:1"]))
        encoder = msgspec.msgpack.Encoder()
        client._client.post = AsyncMock(
            return_value=httpx.Response(
                200,
                content=encoder.encode(ValueEvaluationResponse(values=[[0.1, 0.2], [0.3]], version=7)),
                request=httpx.Request("POST", "http://eval:1/evaluate"),
            )
        )

        await client.evaluate([[1, 2], [3]])
        metrics = client.metrics()

        assert metrics["value/evaluator_requests"] == 1
        assert metrics["value/evaluator_sequences"] == 2
        assert metrics["value/evaluator_tokens"] == 3
        assert metrics["value/evaluator_errors"] == 0
        assert metrics["value/evaluator_error_rate"] == 0
        assert metrics["value/evaluator_latency_seconds_mean"] > 0
        assert metrics["value/evaluator_latency_seconds_max"] > 0
        assert metrics["value/evaluator_response_version_latest"] == 7
        assert metrics["value/evaluator_version_watermark"] == -1
        await client.close()

    asyncio.run(run_test())


def test_evaluate_rejects_oversized_request_before_http():
    async def run_test() -> None:
        client = ValueEvaluatorClient(
            ValueEvaluatorConfig(
                base_url=["http://eval:1"],
                max_pending_tokens=4,
            )
        )
        client._client.post = AsyncMock()

        with pytest.raises(ValueError, match="5 tokens across 2 sequences"):
            await client.evaluate([[1, 2, 3], [4, 5]])

        client._client.post.assert_not_awaited()
        metrics = client.metrics()
        assert metrics["value/evaluator_requests"] == 1
        assert metrics["value/evaluator_sequences"] == 2
        assert metrics["value/evaluator_tokens"] == 5
        assert metrics["value/evaluator_errors"] == 1
        await client.close()

    asyncio.run(run_test())
