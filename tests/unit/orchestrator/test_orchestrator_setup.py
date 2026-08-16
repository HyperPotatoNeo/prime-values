import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from renderers import DefaultRenderer, Qwen3VLRendererConfig

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


def test_group_value_context_rejects_opaque_renderer_before_pool_startup():
    async def run() -> None:
        config = SimpleNamespace(
            model=SimpleNamespace(client=SimpleNamespace(), name="policy-model"),
            renderer=SimpleNamespace(),
            pool_size=None,
            any_policy_sourced=True,
            value_function=SimpleNamespace(privileged_context="group_leave_one_out"),
        )
        renderer = object.__new__(DefaultRenderer)

        with (
            patch("renderers.base.create_renderer", return_value=renderer),
            patch("prime_rl.orchestrator.utils.setup_inference_pool", new=AsyncMock()) as setup_pool_mock,
            pytest.raises(ValueError, match="requires a typed renderer"),
        ):
            await setup_policy_inference_pool(config=config, tokenizer=object())

        setup_pool_mock.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("shipped_version", "live_version", "warmup_updates", "expected", "queries_live_version"),
    [
        pytest.param(0, 1, 1, False, False, id="stale-shipped-version-wins"),
        pytest.param(1, 0, 1, True, False, id="fresh-shipped-version-wins"),
        pytest.param(0, 1, 0, True, False, id="explicit-zero-disables-warmup"),
        pytest.param(None, 1, 1, True, True, id="unscored-batch-uses-live-version"),
        pytest.param(None, 0, 1, False, True, id="unscored-stale-live-version-blocks"),
    ],
)
def test_value_warmup_uses_shipped_provenance(
    shipped_version: int | None,
    live_version: int,
    warmup_updates: int,
    expected: bool,
    queries_live_version: bool,
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
        if queries_live_version:
            orchestrator.value_evaluator.version.assert_awaited_once_with()
        else:
            orchestrator.value_evaluator.version.assert_not_awaited()

    asyncio.run(run())


def test_start_cleans_up_when_setup_fails():
    async def run() -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.config = SimpleNamespace(max_steps=None)
        orchestrator.progress = SimpleNamespace(step=1)
        orchestrator.ckpt_manager = None
        orchestrator.monitor = None
        orchestrator.setup = AsyncMock(side_effect=RuntimeError("setup failed"))
        orchestrator.stop = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        with pytest.raises(RuntimeError, match="setup failed"):
            await orchestrator.start()

        orchestrator.stop.assert_awaited_once_with()

    asyncio.run(run())


def test_start_preserves_main_loop_error_when_final_summary_fails():
    async def run() -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.config = SimpleNamespace(max_steps=None)
        orchestrator.progress = SimpleNamespace(step=1)
        orchestrator.ckpt_manager = None
        orchestrator.setup = AsyncMock()
        orchestrator.lag_monitor = SimpleNamespace(run=AsyncMock())
        orchestrator.periodic_logger = SimpleNamespace(start=AsyncMock())
        orchestrator.dispatcher = SimpleNamespace(start=AsyncMock())
        orchestrator.watcher = SimpleNamespace(start=AsyncMock())
        orchestrator.maybe_trigger_eval = MagicMock()
        orchestrator.main_loop = AsyncMock(side_effect=RuntimeError("main loop failed"))
        orchestrator.monitor = SimpleNamespace(save_final_summary=MagicMock(side_effect=RuntimeError("summary failed")))
        orchestrator.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="main loop failed"):
            await orchestrator.start()

        orchestrator.stop.assert_awaited_once_with()

    asyncio.run(run())


def test_empty_accounting_flush_with_batch_progress_does_not_count_as_stalled():
    async def run() -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.config = SimpleNamespace()
        orchestrator.progress = SimpleNamespace(step=1)
        orchestrator.last_batch_at = None
        orchestrator.consecutive_empty_batches = 4
        orchestrator.train_sink = SimpleNamespace(
            batch_progress=lambda: (3, 8, "rollouts"),
            reset_pre_filter_stats=MagicMock(),
        )
        batch = SimpleNamespace(
            samples=[],
            rollouts=[object()],
            empty_batch_made_progress=True,
        )

        await orchestrator.finalize_train_batch(batch)

        assert orchestrator.consecutive_empty_batches == 0
        orchestrator.train_sink.reset_pre_filter_stats.assert_called_once_with()

    asyncio.run(run())


def test_stop_continues_after_a_cleanup_failure():
    async def run() -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        sender = SimpleNamespace(close=MagicMock(side_effect=RuntimeError("sender failed")))
        monitor = SimpleNamespace(close=MagicMock())
        orchestrator.sender = sender
        orchestrator.dispatcher = None
        orchestrator.train_sink = None
        orchestrator.value_publisher = None
        orchestrator.watcher = None
        orchestrator.periodic_logger = None
        orchestrator.lag_task = None
        orchestrator.component_tasks = []
        orchestrator.inference_metrics = None
        orchestrator.policy_inference = None
        orchestrator.value_evaluator = None
        orchestrator.train_envs = None
        orchestrator.eval_envs = None
        orchestrator.usage_reporter = None
        orchestrator.monitor = monitor

        with pytest.raises(ExceptionGroup, match="orchestrator cleanup failed"):
            await orchestrator.stop()

        sender.close.assert_called_once_with()
        monitor.close.assert_called_once_with()

    asyncio.run(run())
