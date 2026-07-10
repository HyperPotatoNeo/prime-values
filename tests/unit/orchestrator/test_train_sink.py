import asyncio

from prime_rl.orchestrator.train_sink import TrainSink


def test_stop_cancels_scoring_for_incomplete_groups():
    async def run_test() -> None:
        started = asyncio.Event()

        async def score() -> None:
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(score())
        await started.wait()

        sink = TrainSink.__new__(TrainSink)
        sink.scoring_tasks = {1: task}
        await sink.stop()

        assert task.cancelled()
        assert sink.scoring_tasks == {}

    asyncio.run(run_test())
