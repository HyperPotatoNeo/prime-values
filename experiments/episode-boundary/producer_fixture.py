"""Deterministic task/legacy rubric for the real producer protocol smoke."""

import verifiers as legacy
import verifiers.v1 as vf
from datasets import Dataset


class SmokeData(vf.TaskData):
    answer: str
    value_function_prompt: str


class SmokeTask(vf.Task[SmokeData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def exact(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == self.data.answer)


class SmokeTaskset(vf.Taskset[SmokeTask]):
    def load(self) -> list[SmokeTask]:
        return [SmokeTask(SmokeData(idx=0, prompt="seed", answer="seed", value_function_prompt="privileged seed"))]


def load_environment(group_scoring: bool = False) -> legacy.Environment:
    dataset = Dataset.from_list([{"question": "seed", "answer": "seed"}])

    def exact(completion, **kwargs) -> float:
        return float(completion[-1]["content"] == "seed")

    def joint(completions, **kwargs) -> list[float]:
        assert all(completion[-1]["content"] == "seed" for completion in completions)
        return [float(len(completions) + index) for index in range(len(completions))]

    return legacy.SingleTurnEnv(dataset=dataset, rubric=legacy.Rubric(funcs=[joint if group_scoring else exact]))


__all__ = ["SmokeTaskset"]
