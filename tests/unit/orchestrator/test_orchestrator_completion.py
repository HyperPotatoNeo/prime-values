"""Final shipment closes scheduling, but never tears down a required receiver."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.orchestrator import Orchestrator


class ArrivalWindow(list):
    @property
    def effective(self):
        return self

    @property
    def rollouts(self):
        return self

    @property
    def metrics(self):
        return SimpleNamespace(to_wandb=lambda **kwargs: {})

    def by_env(self):
        return {}


def test_final_shipment_drains_eval_and_waits_for_required_nccl_adoption(tmp_path):
    async def run():
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.config = SimpleNamespace(
            max_steps=3, output_dir=tmp_path, weight_broadcast=SimpleNamespace(type="nccl")
        )
        orchestrator.progress = SimpleNamespace(step=3, total_tokens=0, total_samples=0, total_problems=0)
        orchestrator.policy = SimpleNamespace(version=0)
        orchestrator.draining = False
        orchestrator.stopped = asyncio.Event()
        orchestrator.last_batch_at = None
        orchestrator.gate_closed_at = None
        orchestrator.wait_for_policy_time = 0.0
        orchestrator.sender = SimpleNamespace(send=AsyncMock())
        orchestrator.maybe_save_ckpt = AsyncMock(return_value=0.0)
        orchestrator.monitor = MagicMock()
        orchestrator.value_evaluator = None
        orchestrator.value_publisher = None
        orchestrator.usage_reporter = None
        orchestrator.heart = None
        orchestrator.log_train_batch = MagicMock()
        orchestrator.train_sink = SimpleNamespace(
            pre_filter_seen=0, reset_pre_filter_stats=MagicMock(), add=AsyncMock()
        )
        dispatcher = RolloutDispatcher.__new__(RolloutDispatcher)
        dispatcher.out_q = asyncio.Queue()
        dispatcher.dispatch_allowed = asyncio.Event()
        dispatcher.dispatch_allowed.set()
        dispatcher.train_scheduling_disabled = False
        dispatcher.eval_source = None
        dispatcher.metrics = MagicMock()
        blocker = asyncio.Event()
        training = asyncio.create_task(blocker.wait())
        evaluation = asyncio.create_task(blocker.wait())
        await asyncio.sleep(0)
        dispatcher.inflight = {
            training: SimpleNamespace(kind="train", rollout_count=2, env_name="train", group_id="train"),
            evaluation: SimpleNamespace(kind="eval", rollout_count=1, env_name="eval", group_id="eval"),
        }
        dispatcher.groups = {
            "train": SimpleNamespace(kind="train", rollouts_to_schedule=1),
            "eval": SimpleNamespace(kind="eval", rollouts_to_schedule=1),
        }
        dispatcher.inflight_permits = 3
        orchestrator.dispatcher = dispatcher
        orchestrator.maybe_trigger_eval = MagicMock()
        rollout = SimpleNamespace(
            is_trainable=True,
            num_total_tokens=7,
            num_input_tokens=5,
            num_output_tokens=2,
            group_id="completed",
            value_version=None,
            reward=1.0,
            scalar_advantage=lambda: 1.0,
        )
        batch = SimpleNamespace(samples=[object()], rollouts=ArrivalWindow([rollout]))
        with (
            patch("prime_rl.orchestrator.orchestrator.public_rollout_record", return_value={}),
            patch("prime_rl.orchestrator.orchestrator.save_rollouts"),
            patch("prime_rl.orchestrator.orchestrator.trim_process_memory"),
        ):
            await orchestrator.finalize_train_batch(batch)
        assert orchestrator.progress.step == 4
        assert orchestrator.progress.total_tokens == 7
        assert orchestrator.draining and dispatcher.train_scheduling_disabled
        assert not dispatcher.dispatch_allowed.is_set()
        assert training.cancelled() and not evaluation.done()
        assert dispatcher.inflight_permits == 1
        assert set(dispatcher.groups) == {"eval"}
        orchestrator.sender.send.assert_awaited_once()
        assert orchestrator.sender.send.await_args.args[0].step == 3
        orchestrator.maybe_trigger_eval.assert_called_once_with(4)

        # The remaining eval is independent of the closed train gate. Its final
        # result must be consumed, then an idle queue must still keep NCCL alive.
        evaluation.cancel()
        await asyncio.gather(evaluation, return_exceptions=True)
        dispatcher.inflight.pop(evaluation)
        dispatcher.release(1)
        dispatcher.groups.clear()
        eval_rollout = SimpleNamespace(kind="eval")
        dispatcher.out_q.put_nowait(eval_rollout)
        eval_batch = object()
        orchestrator.eval_sink = SimpleNamespace(add=MagicMock(return_value=eval_batch))
        eval_done = asyncio.Event()

        async def finish_eval(received):
            assert received is eval_batch
            eval_done.set()

        orchestrator.finalize_eval_batch = finish_eval
        loop = asyncio.create_task(orchestrator.main_loop())
        try:
            await asyncio.wait_for(eval_done.wait(), timeout=2)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(loop), timeout=0.6)
            assert dispatcher.is_idle and not orchestrator.stopped.is_set()
            assert dispatcher.inflight_permits == 0
            orchestrator.policy.version = 1  # watcher publishes only after receive succeeds
            await asyncio.wait_for(loop, timeout=2)
            assert orchestrator.stopped.is_set()
            orchestrator.train_sink.add.assert_not_awaited()
            orchestrator.sender.send.assert_awaited_once()
        finally:
            if not loop.done():
                loop.cancel()
                await asyncio.gather(loop, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("transport", "max_steps", "version", "ready"),
    [
        ("nccl", 1, 0, True),
        ("nccl", 2, 0, True),
        ("nccl", 3, 0, False),
        ("nccl", 3, 1, True),
        ("nccl", 4, 3, True),
        ("filesystem", 3, 0, True),
    ],
)
def test_completion_requires_only_broadcasts_the_trainer_actually_sends(transport, max_steps, version, ready):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = SimpleNamespace(max_steps=max_steps, weight_broadcast=SimpleNamespace(type=transport))
    orchestrator.policy = SimpleNamespace(version=version)
    assert orchestrator._required_broadcasts_adopted() is ready
