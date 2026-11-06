"""Native vLLM token serving with the pinned renderer's response representation.

Upstream owns request handling, DP routing, generation limits and streaming.
Only final token responses are adapted: literal token labels without changing
logprob numbers, compact routed-expert bytes, and cached-prefix usage details.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateTokensChoice,
    GenerateTokensResponse,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import ErrorResponse, PromptTokenUsageInfo, UsageInfo
from vllm.outputs import RequestOutput

from prime_rl.inference.vllm.routed_experts import RoutedExpertsCapture


class PrimeRlGenerateResponseChoice(GenerateTokensChoice):
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]
    logprobs: ChatCompletionLogProbs | None = None  # type: ignore[assignment]


class PrimeRlGenerateResponse(GenerateTokensResponse):
    choices: list[PrimeRlGenerateResponseChoice]


def _renderer_logprobs(choice: GenerateTokensChoice) -> ChatCompletionLogProbs | None:
    if choice.logprobs is None:
        return None
    rows = choice.logprobs.content
    if rows is None:
        return ChatCompletionLogProbs(content=None)
    token_ids = choice.token_ids or []
    if len(rows) != len(token_ids):
        raise ValueError("Sampled token logprobs must align with response token_ids")
    content = []
    for token_id, row in zip(token_ids, rows, strict=True):
        if row.token_id != token_id:
            raise ValueError("Sampled logprob token identity does not match response token_ids")
        content.append(
            ChatCompletionLogProbsContent(
                token=f"token_id:{token_id}",
                logprob=row.logprob,
                bytes=None,
                top_logprobs=[
                    ChatCompletionLogProb(token=f"token_id:{entry.token_id}", logprob=entry.logprob, bytes=None)
                    for entry in row.top_logprobs
                ],
            )
        )
    return ChatCompletionLogProbs(content=content)


class _GenerateRoutedExpertsCapture(RoutedExpertsCapture):
    def post_process(self, response: GenerateTokensResponse) -> PrimeRlGenerateResponse:
        choices = [
            PrimeRlGenerateResponseChoice(
                **choice.model_dump(exclude={"routed_experts", "logprobs"}),
                routed_experts=self.routed_experts.get(choice.index),
                logprobs=_renderer_logprobs(choice),
            )
            for choice in response.choices
        ]
        return PrimeRlGenerateResponse(**{**response.model_dump(exclude={"choices"}), "choices": choices})


class _FinalOutputCapture:
    """Wraps a ``RequestOutput`` async generator to record the last yielded item.

    Needed so the response builder can construct a ``usage`` block from
    ``final_res.prompt_token_ids`` / ``output.token_ids`` / ``num_cached_tokens``
    after delegating iteration to upstream.
    """

    def __init__(self, source: AsyncIterable[RequestOutput]) -> None:
        # ``source`` may be any async-iterable — including
        # ``_GenerateRoutedExpertsCapture``, which exposes the protocol via
        # ``async def __aiter__`` (an async generator function) and has no
        # ``__anext__`` method. Drive it through ``async for`` rather than
        # poking ``__anext__`` directly so both shapes work.
        self._source = source
        self.final_res: RequestOutput | None = None

    async def __aiter__(self) -> AsyncGenerator[RequestOutput, None]:
        async for item in self._source:
            self.final_res = item
            yield item


def _build_usage(final_res: RequestOutput) -> UsageInfo:
    assert final_res.prompt_token_ids is not None
    num_prompt_tokens = len(final_res.prompt_token_ids)
    if final_res.encoder_prompt_token_ids is not None:
        num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    # Always emit cached tokens when vLLM reports any. Upstream gates this on
    # ``enable_prompt_tokens_details`` (default False) for OpenAI-API compat,
    # but ``/inference/v1/generate`` is prime-rl internal — the cache-discount
    # billing pipeline always wants the cached subset surfaced.
    if final_res.num_cached_tokens:
        usage.prompt_tokens_details = PromptTokenUsageInfo(cached_tokens=final_res.num_cached_tokens)
    return usage


class PrimeRlServingTokens(ServingTokens):
    """Preserve the renderer wire contract while upstream handles generation."""

    async def serve_tokens_full_generator(
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse:
        capture = _GenerateRoutedExpertsCapture(
            result_generator, start=request.sampling_params.routed_experts_prompt_start
        )
        final_capture = _FinalOutputCapture(capture)
        response = await super().serve_tokens_full_generator(
            request, final_capture, request_id, model_name, request_metadata
        )
        if not isinstance(response, GenerateTokensResponse):
            return response
        response = capture.post_process(response)
        # Engine-global labels describe full base-model reloads. Adapter weights
        # have a separate lifecycle and no validated version protocol here.
        if request.model in self.models.lora_requests or model_name in self.models.lora_requests:
            for choice in response.choices:
                choice.weight_versions = None
        if final_capture.final_res is not None:
            response.usage = _build_usage(final_capture.final_res)
        return response
