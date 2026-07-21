import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from renderers import Qwen3VLRendererConfig

from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.orchestrator.utils import setup_policy_inference_pool


def test_setup_policy_inference_pool_uses_renderer_when_enabled():
    async def run() -> None:
        tokenizer = object()
        renderer_settings = Qwen3VLRendererConfig()
        config = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                name="policy-model",
            ),
            renderer=renderer_settings,
            pool_size=None,
            any_policy_sourced=True,
        )
        renderer = object()
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_policy_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_awaited_once_with(
            config.model.client,
            model_name="policy-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
            pool_size=None,
        )

    asyncio.run(run())


def test_setup_policy_inference_pool_keeps_renderer_without_policy_sampling():
    """Frozen-sourced runs (e.g. sft) have no train env sampling from the live
    policy, but training is renderer-only: the renderer is still built and the
    pool is wired with the renderer train client. ``any_policy_sourced`` only
    flips the log line, not the pool setup."""

    async def run() -> None:
        tokenizer = object()
        renderer_settings = Qwen3VLRendererConfig()
        config = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                name="policy-model",
            ),
            renderer=renderer_settings,
            pool_size=None,
            any_policy_sourced=False,
        )
        renderer = object()
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_policy_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_awaited_once_with(
            config.model.client,
            model_name="policy-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
            pool_size=None,
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("shipped_version", "live_version", "warmup_updates", "expected"),
    [
        pytest.param(0, 1, 1, False, id="stale-shipped-version-wins"),
        pytest.param(1, 0, 1, True, id="fresh-shipped-version-wins"),
        pytest.param(0, 1, 0, True, id="explicit-zero-disables-warmup"),
        pytest.param(None, 1, 1, True, id="unscored-batch-uses-live-version"),
        pytest.param(None, 0, 1, False, id="unscored-stale-live-version-blocks"),
    ],
)
def test_value_warmup_uses_shipped_provenance(
    shipped_version: int | None,
    live_version: int,
    warmup_updates: int,
    expected: bool,
):
    async def run() -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.config = SimpleNamespace(
            value_function=SimpleNamespace(warmup_updates=warmup_updates),
        )
        orchestrator.value_evaluator = SimpleNamespace(
            version=AsyncMock(return_value=live_version),
        )
        orchestrator.last_warmup_value_version = None
        batch = SimpleNamespace(shipped_value_version_min=shipped_version)

        assert await orchestrator._passes_value_warmup(batch) is expected
        orchestrator.value_evaluator.version.assert_awaited_once_with()

    asyncio.run(run())
