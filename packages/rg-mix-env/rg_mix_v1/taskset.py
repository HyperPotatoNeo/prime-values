"""Native cached RG mix tasks; selected data owns all private scoring inputs."""

from functools import cached_property
from pathlib import Path
from typing import Literal

import reasoning_gym as rg
import verifiers.v1 as vf
from pydantic import Field, JsonValue
from rg_mix_shared import DEFAULT_SYSTEM_PROMPT, TASK_VARIANTS, _load_dataset

from rg_mix_v1.scoring import format_reward, response_for_scoring

_VARIANTS = {variant["id"]: variant for variant in TASK_VARIANTS}


class RGMixData(vf.TaskData):
    value_function_prompt: str | None = None
    """Optional critic-only prefix; preserved on the worker wire and redacted in public records."""

    answer: str
    """Global scorer index, distinct from the split-local task idx."""
    variant_id: str
    scorer_entry: dict[str, JsonValue]
    """Private reward data; never appended to the policy prompt or value context."""


class RGMixTask(vf.Task[RGMixData]):
    @cached_property
    def _scorer(self):
        variant = _VARIANTS[self.data.variant_id]
        return rg.create_dataset(variant["task"], seed=1, size=1, **variant["config"])

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.info["variant_id"] = self.data.variant_id

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def check_answer_reward_func(self, trace: vf.Trace) -> float:
        response = response_for_scoring(trace.assistant_messages)
        score = self._scorer.score_answer(answer=response, entry=self.data.scorer_entry)
        trace.record_metric("check_answer_reward_func", score)
        return score

    @vf.reward(weight=0.0)
    async def format_reward_func(self, trace: vf.Trace) -> float:
        score = format_reward(trace.assistant_messages)
        trace.record_metric("format_reward_func", score)
        return score


class RGMixConfig(vf.TasksetConfig):
    dataset_path: Path
    """Existing generated dataset directory, including metadata.json."""
    num_train_examples: int = Field(10000, ge=0)
    num_eval_examples: int = Field(2048, ge=0)
    seed: int = 42
    """Accepted for configuration compatibility; unused when selecting cached rows."""
    split: Literal["train", "eval"] = "train"
    prompt_text: str = DEFAULT_SYSTEM_PROMPT
    """Inline RG system prompt. The inherited system_prompt is an optional file override."""


class RGMixTaskset(vf.Taskset[RGMixTask, RGMixConfig]):
    def load(self) -> list[RGMixTask]:
        rows, entry_map, entries_cache, _ = _load_dataset(self.config.dataset_path)
        if self.config.split == "train":
            selected = rows[: self.config.num_train_examples]
        else:
            start = self.config.num_train_examples
            selected = rows[start : start + self.config.num_eval_examples]
        tasks = []
        for idx, row in enumerate(selected):
            answer = str(row["answer"])
            global_idx = int(answer)
            variant_id, _ = entry_map[global_idx]
            if row["task"] != variant_id:
                raise ValueError(f"RG row {global_idx} variant does not match its scoring metadata")
            data = RGMixData(
                idx=idx,
                prompt=row["question"],
                system_prompt=self.config.prompt_text,
                answer=answer,
                variant_id=variant_id,
                scorer_entry=entries_cache[global_idx],
            )
            tasks.append(RGMixTask(data, self.config.task))
        return tasks
