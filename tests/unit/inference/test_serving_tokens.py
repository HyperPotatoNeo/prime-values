"""PrimeRL response compatibility on genuine vLLM token protocol objects."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
from renderers.client import _parse_completion_logprobs
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateLogProb,
    GenerateLogProbs,
    GenerateLogProbsContent,
    GenerateRequest,
    GenerateTokensChoice,
    GenerateTokensResponse,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.exceptions import VLLMValidationError
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput, WeightVersionSpan
from vllm.renderers.params import TokenizeParams

from prime_rl.inference.vllm.routed_experts import serialize_routed_experts
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlGenerateResponse,
    PrimeRlGenerateResponseChoice,
    PrimeRlServingTokens,
    _build_usage,
    _FinalOutputCapture,
    _GenerateRoutedExpertsCapture,
    _renderer_logprobs,
)


def _decode_routed_experts(encoded: dict) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode_as_bytearray(encoded["data"]),
        dtype=np.uint8,
    ).reshape(encoded["shape"])


async def _empty_request_outputs():
    if False:
        yield


def test_subclass_delegates_requests_but_adapts_final_responses():
    assert PrimeRlServingTokens.serve_tokens is PrimeRlServingTokens.__mro__[1].serve_tokens
    assert (
        PrimeRlServingTokens.serve_tokens_full_generator
        is not PrimeRlServingTokens.__mro__[1].serve_tokens_full_generator
    )


@pytest.mark.parametrize(
    ("request_model", "resolved_model", "expected_version"),
    [("base", "base", "7"), ("adapter", "base", None), ("base", "adapter", None)],
)
def test_real_parent_final_response_keeps_base_labels_but_omits_adapter_labels(
    request_model, resolved_model, expected_version
):
    # Exercise the real upstream final generator without an engine: only the
    # model registry and two logging flags are needed for this token response.
    serving = object.__new__(PrimeRlServingTokens)
    serving.models = SimpleNamespace(lora_requests={"adapter": object()})
    serving.enable_prompt_tokens_details = False
    serving.enable_log_outputs = False
    sampled_logprob = -0.12345678912345
    output = CompletionOutput(
        index=0,
        text="",
        token_ids=[17],
        cumulative_logprob=sampled_logprob,
        logprobs=[{17: Logprob(sampled_logprob, rank=1)}],
        finish_reason="length",
        weight_versions=[WeightVersionSpan(version="7", start=0, end=1)],
    )
    final = RequestOutput(
        request_id="req",
        prompt=None,
        prompt_token_ids=[5, 11],
        prompt_logprobs=None,
        outputs=[output],
        finished=True,
        num_cached_tokens=1,
    )

    async def result_generator():
        yield final

    request = GenerateRequest.model_validate(
        {"model": request_model, "token_ids": [5, 11], "sampling_params": {"logprobs": 1, "max_tokens": 1}}
    )
    response = asyncio.run(
        serving.serve_tokens_full_generator(
            request, result_generator(), "req", resolved_model, RequestResponseMetadata(request_id="req")
        )
    )
    assert isinstance(response, PrimeRlGenerateResponse)
    choice = response.choices[0]
    assert choice.token_ids == [17]
    assert choice.logprobs.content[0].logprob == sampled_logprob
    assert response.usage.prompt_tokens_details.cached_tokens == 1
    assert response.usage.total_tokens == 3
    if expected_version is None:
        assert choice.weight_versions is None
    else:
        assert [(span.version, span.start, span.end) for span in choice.weight_versions] == [(expected_version, 0, 1)]
    assert output.weight_versions == [WeightVersionSpan(version="7", start=0, end=1)]


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
    response = GenerateTokensResponse(
        request_id="request-id",
        choices=[
            GenerateTokensChoice(
                index=0,
                token_ids=[1, 2, 3],
                routed_experts="upstream-npy-payload",
            )
        ],
    )

    processed = capture.post_process(response)

    assert processed.choices[0].routed_experts == compact_routed_experts


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
    # The response bridge retains upstream usage fields for router accounting.
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
        "completion_tokens_details": None,
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


def test_upstream_request_preserves_explicit_generation_budget():
    explicit = GenerateRequest.model_validate({"token_ids": [1, 2], "sampling_params": {"max_tokens": 16}})
    absent = GenerateRequest.model_validate({"token_ids": [1, 2], "sampling_params": {}})
    assert explicit.is_sampling_param_provided("max_tokens")
    assert not absent.is_sampling_param_provided("max_tokens")
    assert explicit.sampling_params.max_tokens == 16


def test_native_chat_merges_default_stop_tokens_without_mutating_requests():
    defaults = {"stop_token_ids": [7, 9]}
    request = ChatCompletionRequest(model="base", messages=[{"role": "user", "content": "hi"}], stop_token_ids=[3, 7])
    params = request.to_sampling_params(max_tokens=8, default_sampling_params=defaults)
    assert params.stop_token_ids == [3, 7, 9]
    assert request.stop_token_ids == [3, 7]
    assert defaults == {"stop_token_ids": [7, 9]}
    request.stop_token_ids = None
    assert request.to_sampling_params(max_tokens=8, default_sampling_params=defaults).stop_token_ids == [7, 9]


def test_native_tokenization_uses_context_budget_and_generation_clamps_separately():
    params = TokenizeParams(max_total_tokens=16, max_output_tokens=8)
    assert params.max_input_tokens == 16
    prompt = {"prompt_token_ids": list(range(12))}
    assert params.apply_post_tokenization(None, prompt)["prompt_token_ids"] == list(range(12))
    assert get_max_tokens(16, 8, 12, {}) == 4
    with pytest.raises(VLLMValidationError, match="maximum context length"):
        params.apply_post_tokenization(None, {"prompt_token_ids": list(range(17))})


def _numeric_response():
    return GenerateTokensResponse(
        request_id="req-exact",
        model="policy",
        created=123,
        choices=[
            GenerateTokensChoice(
                index=0,
                token_ids=[17, 23],
                finish_reason="length",
                stop_reason=99,
                logprobs=GenerateLogProbs(
                    content=[
                        GenerateLogProbsContent(
                            token_id=17,
                            logprob=-0.12345678912345,
                            top_logprobs=[
                                GenerateLogProb(token_id=17, logprob=-0.12345678912345, rank=1),
                                GenerateLogProb(token_id=999, logprob=-3.98765432198765, rank=2),
                            ],
                        ),
                        GenerateLogProbsContent(token_id=23, logprob=-0.0000000000012345),
                    ]
                ),
                weight_versions=[
                    WeightVersionSpan(version="default", start=0, end=1),
                    WeightVersionSpan(version="7", start=1, end=2),
                ],
            )
        ],
        prompt_logprobs=[None, {11: Logprob(logprob=-0.24681357912345, rank=1, decoded_token="prompt")}],
        prompt_token_ids=[5, 11],
        prompt_token_id_logprobs="opaque-prompt-carrier",
        usage=UsageInfo(prompt_tokens=2, completion_tokens=2, total_tokens=4),
        kv_transfer_params={"sentinel": "kv"},
        ec_transfer_params={"sentinel": "encoder"},
    )


def test_full_response_bridge_preserves_exact_numbers_spans_and_base_fields():
    response = _numeric_response()
    before = response.model_dump(mode="python")
    capture = _GenerateRoutedExpertsCapture(_empty_request_outputs())
    compact = {"data": "AQID", "shape": [1, 1, 3], "start": 1}
    capture.routed_experts[0] = compact
    converted = capture.post_process(response)
    assert (
        PrimeRlGenerateResponse.model_validate_json(converted.model_dump_json()).model_dump() == converted.model_dump()
    )
    assert response.model_dump(mode="python") == before
    assert converted.model_dump(exclude={"choices"}) == response.model_dump(exclude={"choices"})
    assert converted.choices[0].model_dump(exclude={"routed_experts", "logprobs"}) == response.choices[0].model_dump(
        exclude={"routed_experts", "logprobs"}
    )
    choice = converted.model_dump(mode="json")["choices"][0]
    assert choice["routed_experts"] == compact
    assert _parse_completion_logprobs(choice, [17, 23]) == [-0.12345678912345, -0.0000000000012345]
    assert [x["token"] for x in choice["logprobs"]["content"]] == ["token_id:17", "token_id:23"]
    assert [x["token"] for x in choice["logprobs"]["content"][0]["top_logprobs"]] == ["token_id:17", "token_id:999"]
    assert [x["logprob"] for x in choice["logprobs"]["content"][0]["top_logprobs"]] == [
        -0.12345678912345,
        -3.98765432198765,
    ]
    assert "completion_logprobs" not in choice
    assert "token_id" not in choice["logprobs"]["content"][0]


@pytest.mark.parametrize("corruption", ["identity", "count"])
def test_response_bridge_rejects_misaligned_sampled_evidence(corruption):
    choice = _numeric_response().choices[0]
    if corruption == "identity":
        choice.logprobs.content[0].token_id = 99
    else:
        choice.logprobs.content.pop()
    with pytest.raises(ValueError, match="token"):
        _renderer_logprobs(choice)


def test_response_bridge_preserves_absent_and_empty_logprobs():
    assert _renderer_logprobs(GenerateTokensChoice(index=0, token_ids=[])) is None
    for token_ids in [[], None]:
        empty = _renderer_logprobs(
            GenerateTokensChoice(index=0, token_ids=token_ids, logprobs=GenerateLogProbs(content=[]))
        )
        assert empty.content == []
