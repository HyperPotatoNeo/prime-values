"""Sanity tests for the prime-RL ``ServingTokens`` subclass.

The full happy-path is owned upstream by vLLM 0.20's
``vllm/entrypoints/serve/disagg`` test suite. We only cover the prime-RL
deltas here:
    * ``serialize_routed_experts`` round-trips a compact raw-byte payload.
    * The subclass attaches its overrides without monkey-patching the parent.
    * ``_client_set_max_tokens`` distinguishes raw-body shapes correctly.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pybase64
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.serve.disagg.protocol import GenerateResponse, GenerateResponseChoice

from prime_rl.inference.vllm.routed_experts import serialize_routed_experts
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlGenerateResponse,
    PrimeRlGenerateResponseChoice,
    PrimeRlServingTokens,
    _build_usage,
    _client_set_max_tokens,
    _compact_sampler_head,
    _configure_token_output,
    _FinalOutputCapture,
    _GenerateRoutedExpertsCapture,
)


def test_token_output_bookkeeping_preserves_sampling_and_text_stops():
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    for kwargs, stream, expected_detokenize in [
        ({}, False, False),
        ({"stop": ["END"]}, False, True),
        ({"prompt_logprobs": 1}, False, True),
        ({}, True, True),
    ]:
        params = SamplingParams(seed=42, temperature=1, logprobs=128, max_tokens=8192, **kwargs)
        before = (params.seed, params.temperature, params.logprobs, params.max_tokens, params.stop)
        _configure_token_output(params, stream=stream, force_no_detokenize=False)
        assert params.detokenize is expected_detokenize
        assert params.output_kind == (RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY)
        assert before == (params.seed, params.temperature, params.logprobs, params.max_tokens, params.stop)


def test_compact_head_preserves_sampler_and_isolates_concurrent_requests():
    from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
    from vllm.entrypoints.serve.disagg.protocol import GenerateRequest
    from vllm.logprobs import Logprob
    from vllm.outputs import CompletionOutput, RequestOutput
    from vllm.sampling_params import SamplingParams

    from prime_rl.inference.vllm.score_centering import encode_sampler_head

    server = object.__new__(PrimeRlServingTokens)
    server.model_config = SimpleNamespace(enable_return_routed_experts=False)
    server.enable_prompt_tokens_details = False
    server.enable_log_outputs = False
    row = {i: Logprob(logprob=-5.0 - i / 100, rank=i + 1) for i in range(128)}
    row[999] = Logprob(logprob=-12.0, rank=1000)
    output = CompletionOutput(
        index=0, text="", token_ids=[0, 999], cumulative_logprob=-17.0, logprobs=[row, row], finish_reason="length"
    )
    final = RequestOutput(
        request_id="test", prompt=None, prompt_token_ids=[1], prompt_logprobs=None, outputs=[output], finished=True
    )

    async def respond(compact):
        params = SamplingParams(logprobs=128, extra_args={"score_centering_top_k": 128} if compact else None)
        request = GenerateRequest(token_ids=[1], sampling_params=params)

        async def source():
            await asyncio.sleep(0)
            assert params.logprobs == 128
            yield final

        result = await server.serve_tokens_full_generator(
            request, source(), "test", "model", RequestResponseMetadata(request_id="test")
        )
        assert not _compact_sampler_head.get()
        assert params.logprobs == 128
        return result

    async def both():
        return await asyncio.gather(respond(True), respond(False), respond(True))

    compact, regular, compact2 = asyncio.run(both())
    assert compact.model_dump(exclude={"request_id"}) == compact2.model_dump(exclude={"request_id"})
    assert compact.choices[0].sampler_head == encode_sampler_head(output, 128)
    assert regular.choices[0].sampler_head is None
    for small, full in zip(compact.choices[0].logprobs.content, regular.choices[0].logprobs.content):
        assert small.token == full.token and small.logprob == full.logprob
        assert small.top_logprobs == [] and len(full.top_logprobs) == 128
    assert compact.choices[0].logprobs.content[1].logprob == -12.0


def _decode_routed_experts(encoded: dict) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode_as_bytearray(encoded["data"]),
        dtype=np.uint8,
    ).reshape(encoded["shape"])


class _FakeRawRequest:
    def __init__(self, body):
        self._body = body
        self._raise = isinstance(body, Exception)

    async def json(self):
        if self._raise:
            raise self._body
        return self._body


async def _empty_request_outputs():
    if False:
        yield


def test_subclass_only_overrides_serve_tokens():
    assert PrimeRlServingTokens.serve_tokens is not PrimeRlServingTokens.__mro__[1].serve_tokens
    assert (
        PrimeRlServingTokens.serve_tokens_full_generator
        is not PrimeRlServingTokens.__mro__[1].serve_tokens_full_generator
    )


def test_serialize_routed_experts_uses_compact_raw_payload():
    routed_experts = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int64,
    )

    encoded = serialize_routed_experts(routed_experts)
    assert encoded is not None

    decoded = _decode_routed_experts(encoded)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, routed_experts)


def test_generate_response_post_process_replaces_upstream_routed_experts():
    compact_routed_experts = {"data": "AQID", "shape": [1, 1, 3], "start": 0}
    capture = _GenerateRoutedExpertsCapture(_empty_request_outputs())
    capture.routed_experts[0] = compact_routed_experts
    response = GenerateResponse(
        request_id="request-id",
        choices=[
            GenerateResponseChoice(
                index=0,
                token_ids=[1, 2, 3],
                routed_experts="upstream-npy-payload",
            )
        ],
    )

    processed = capture.post_process(response)

    assert processed.choices[0].routed_experts == compact_routed_experts


def test_client_set_max_tokens_recognizes_explicit_value():
    body = {"token_ids": [1, 2, 3], "sampling_params": {"max_tokens": 256}}
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(body))) is True


def test_client_set_max_tokens_detects_unset():
    body = {"token_ids": [1, 2, 3], "sampling_params": {}}
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(body))) is False

    body_without_sp = {"token_ids": [1, 2, 3]}
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(body_without_sp))) is False


class _FakeOutput:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class _FakeRequestOutput:
    """Minimal stand-in for ``vllm.outputs.RequestOutput``.

    ``_build_usage`` only touches four attributes; constructing a real
    ``RequestOutput`` would require a full ``CompletionOutput`` graph and
    isn't worth it for a serialization-shape test.
    """

    def __init__(self, prompt_token_ids, output_token_ids_list, num_cached_tokens=0, encoder_prompt_token_ids=None):
        self.prompt_token_ids = prompt_token_ids
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.outputs = [_FakeOutput(t) for t in output_token_ids_list]
        self.num_cached_tokens = num_cached_tokens


def test_prime_rl_generate_response_serializes_usage_block():
    # Regression for prime-rl PR #2408: parent ``GenerateResponse`` doesn't
    # declare ``usage``, so the field must be declared on the subclass for
    # Pydantic to emit it in JSON. Without this the router can't extract
    # per-run token / cache counts for billing.
    response = PrimeRlGenerateResponse(
        request_id="req-1",
        choices=[PrimeRlGenerateResponseChoice(index=0, token_ids=[1, 2, 3])],
        usage=UsageInfo(prompt_tokens=4, completion_tokens=3, total_tokens=7),
    )
    payload = response.model_dump(mode="json")
    assert payload["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
        "prompt_tokens_details": None,
    }


def test_build_usage_sums_prompt_and_completion_tokens():
    final_res = _FakeRequestOutput(
        prompt_token_ids=[1, 2, 3, 4, 5],
        output_token_ids_list=[[10, 11], [20, 21, 22]],
    )
    usage = _build_usage(final_res)
    assert usage.prompt_tokens == 5
    assert usage.completion_tokens == 5  # 2 + 3
    assert usage.total_tokens == 10
    assert usage.prompt_tokens_details is None


def test_build_usage_includes_encoder_prompt_tokens():
    final_res = _FakeRequestOutput(
        prompt_token_ids=[1, 2, 3],
        output_token_ids_list=[[10]],
        encoder_prompt_token_ids=[100, 101],
    )
    usage = _build_usage(final_res)
    assert usage.prompt_tokens == 5  # 3 + 2
    assert usage.total_tokens == 6


def test_build_usage_reports_cached_tokens_unconditionally():
    # Unlike upstream's ``enable_prompt_tokens_details`` gate, prime-rl always
    # surfaces cached tokens — the cache-discount billing pipeline needs them.
    final_res = _FakeRequestOutput(
        prompt_token_ids=[1, 2, 3, 4],
        output_token_ids_list=[[10, 11]],
        num_cached_tokens=3,
    )
    usage = _build_usage(final_res)
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 3


def test_build_usage_skips_cached_tokens_when_zero():
    # Don't emit a details block with cached=0, which would be misleading
    # to the router's billing extractor.
    final_res = _FakeRequestOutput(
        prompt_token_ids=[1, 2, 3, 4],
        output_token_ids_list=[[10, 11]],
        num_cached_tokens=0,
    )
    usage = _build_usage(final_res)
    assert usage.prompt_tokens_details is None


def test_final_output_capture_records_last_item():
    async def _gen():
        for r in [
            _FakeRequestOutput(prompt_token_ids=[1], output_token_ids_list=[[1]]),
            _FakeRequestOutput(prompt_token_ids=[1, 2], output_token_ids_list=[[1, 2]]),
            _FakeRequestOutput(prompt_token_ids=[1, 2, 3], output_token_ids_list=[[1, 2, 3]]),
        ]:
            yield r

    async def _drain(capture):
        async for _ in capture:
            pass

    capture = _FinalOutputCapture(_gen())
    asyncio.run(_drain(capture))
    assert capture.final_res is not None
    assert capture.final_res.prompt_token_ids == [1, 2, 3]


def test_final_output_capture_works_over_async_def_aiter_source():
    # ``_GenerateRoutedExpertsCapture`` exposes the async-iterator protocol
    # via ``async def __aiter__`` (an async generator function) and has no
    # ``__anext__``. The wrapper must drive it through ``async for`` rather
    # than poking ``__anext__`` directly, or routed-experts runs raise
    # AttributeError before the response is built.

    class _AsyncGenAiterSource:
        def __init__(self, items):
            self._items = items

        async def __aiter__(self):
            for item in self._items:
                yield item

    items = [
        _FakeRequestOutput(prompt_token_ids=[1], output_token_ids_list=[[1]]),
        _FakeRequestOutput(prompt_token_ids=[1, 2], output_token_ids_list=[[1, 2]]),
    ]
    capture = _FinalOutputCapture(_AsyncGenAiterSource(items))

    async def _drain():
        async for _ in capture:
            pass

    asyncio.run(_drain())
    assert capture.final_res is not None
    assert capture.final_res.prompt_token_ids == [1, 2]


def test_final_output_capture_handles_empty_stream():
    capture = _FinalOutputCapture(_empty_request_outputs())

    async def _drain():
        async for _ in capture:
            pass

    asyncio.run(_drain())
    assert capture.final_res is None


def test_client_set_max_tokens_assumes_set_when_body_unreadable():
    # No raw_request → can't tell, don't override.
    assert asyncio.run(_client_set_max_tokens(None)) is True

    # body read raises → can't tell, don't override.
    err = ValueError("bad json")
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(err))) is True

    # non-dict body → can't tell, don't override.
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest([1, 2, 3]))) is True
