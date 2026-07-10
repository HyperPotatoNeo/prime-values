"""RG-Mix: a mixed reasoning-gym single-turn environment.

A verifiers environment over a fixed mixture of five reasoning-gym task variants
(arc_1d, sokoban_hard, countdown_7, zebra_puzzles_7, cryptarithm). Rewards are in
[0, 1] from each variant's own ``score_answer``. Prompts are loaded from a saved
dataset (``dataset_path``) built by ``build_dataset.py`` so every run trains on the
same fixed data. This is the environment used for the value-corrected GRPO
baseline experiments; it matches the offline static-value diagnostics so their
coefficient recommendations transfer.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset, DatasetDict, load_from_disk

try:
    import reasoning_gym as rg
    from reasoning_gym.utils import SYSTEM_PROMPTS
except ImportError as exc:  # pragma: no cover - requires the reasoning-gym dependency.
    raise ImportError(
        "rg_mix requires reasoning-gym; install the 'rg-mix' env or `uv run --with reasoning-gym`."
    ) from exc


TASK_VARIANTS = [
    {"id": "arc_1d", "task": "arc_1d", "config": {}},
    {
        "id": "sokoban_hard",
        "task": "sokoban",
        "config": {"min_boxes": 3, "max_boxes": 4, "max_w": 9, "max_h": 9},
    },
    {"id": "countdown_7", "task": "countdown", "config": {"min_numbers": 7, "max_numbers": 7}},
    {
        "id": "zebra_puzzles_7",
        "task": "zebra_puzzles",
        "config": {"num_people": 7, "num_characteristics": 5},
    },
    {"id": "cryptarithm", "task": "cryptarithm", "config": {}},
]
PASS_AT_1S = {
    "arc_1d": 0.93,
    "sokoban_hard": 0.481,
    "countdown_7": 0.545,
    "zebra_puzzles_7": 0.51,
    "cryptarithm": 0.755,
}
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]
MAX_EXTRACTED_ANSWER_CHARS = 512


def _extract_answer(response: str) -> str | None:
    answer_end = response.rfind("</answer>")
    if answer_end < 0:
        return None
    answer_start = response.rfind("<answer>", 0, answer_end)
    if answer_start < 0:
        return None
    answer = response[answer_start + len("<answer>") : answer_end].strip()
    if len(answer) > MAX_EXTRACTED_ANSWER_CHARS:
        return None
    return answer


def _assistant_texts(completion: vf.Messages | str) -> list[str]:
    if isinstance(completion, str):
        return [completion]
    texts = []
    for message in completion:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "assistant":
            continue
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")) for part in content
            )
        texts.append(str(content or ""))
    return texts


def _extract_answer_text(completion: vf.Messages | str) -> str | None:
    for text in reversed(_assistant_texts(completion)):
        extracted = _extract_answer(text)
        if extracted is not None:
            return extracted
    return None


def _score_answer(ds: Any, response: str | None, entry: dict[str, Any]) -> float:
    if not response:
        return 0.0
    score = float(ds.score_answer(answer=response, entry=entry))
    if score < 0.0 or score > 1.0:
        raise ValueError(f"RGMix reward must be in [0,1], got {score}")
    return score


def _dataset_root(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if (path / "metadata.json").exists() and (path / "dataset").exists():
        return path
    if path.name == "dataset" and (path.parent / "metadata.json").exists():
        return path.parent
    raise ValueError(f"dataset_path must be a saved RGMix root with metadata.json and dataset/: {dataset_path}")


def _format_example(example: dict[str, Any], entry_map: list[tuple[str, int]]) -> dict[str, Any]:
    answer_idx = int(example["answer"])
    vid = entry_map[answer_idx][0]
    return {
        "question": example["question"],
        "answer": str(answer_idx),
        "rg_task": vid,
    }


def _load_saved_dataset(dataset_path: str, num_train_examples: int, num_eval_examples: int):
    root = _dataset_root(dataset_path)
    with (root / "metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    entry_map = [tuple(x) for x in meta["entry_map"]]
    entries_cache = {int(k): v for k, v in meta["entries_cache"].items()}

    hf_ds = load_from_disk(str(root / "dataset"))
    if isinstance(hf_ds, DatasetDict):
        first_split = "train" if "train" in hf_ds else next(iter(hf_ds.keys()))
        hf_ds = hf_ds[first_split]
    rows = [_format_example(dict(row), entry_map) for row in hf_ds]

    total_needed = num_train_examples + num_eval_examples
    if total_needed > len(rows):
        raise ValueError(f"requested {total_needed} RGMix examples, but saved dataset has {len(rows)}")

    variant_datasets = {
        variant["id"]: rg.create_dataset(variant["task"], seed=1, size=1, **variant["config"])
        for variant in TASK_VARIANTS
    }
    train_dataset = Dataset.from_list(rows[:num_train_examples])
    eval_dataset = Dataset.from_list(rows[num_train_examples:total_needed])
    return train_dataset, eval_dataset, entry_map, entries_cache, variant_datasets


class RGMixEnv(vf.SingleTurnEnv):
    def __init__(
        self,
        *,
        num_train_examples: int,
        num_eval_examples: int,
        seed: int,
        dataset_path: str,
    ) -> None:
        random.seed(seed)
        train_dataset, eval_dataset, self._entry_map, self._entries_cache, self._variant_datasets = _load_saved_dataset(
            dataset_path, num_train_examples, num_eval_examples
        )
        parser = vf.XMLParser(fields=["answer"])
        env_ref = self

        def reward_func(completion: vf.Messages | str, answer: Any, **kwargs) -> float:
            del kwargs
            answer_idx = int(answer)
            vid, _ = env_ref._entry_map[answer_idx]
            return _score_answer(
                env_ref._variant_datasets[vid],
                _extract_answer_text(completion),
                env_ref._entries_cache[answer_idx],
            )

        rubric = vf.Rubric(funcs=[reward_func], weights=[1.0], parser=parser)
        rubric.pass_at_1s = PASS_AT_1S
        super().__init__(
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            message_type="chat",
        )

    def score_candidate(self, answer_idx: int, response: str) -> float:
        vid, _ = self._entry_map[answer_idx]
        return _score_answer(self._variant_datasets[vid], _extract_answer(response), self._entries_cache[answer_idx])

    def score_completion(self, answer_idx: int, completion: vf.Messages | str) -> float:
        vid, _ = self._entry_map[answer_idx]
        return _score_answer(
            self._variant_datasets[vid],
            _extract_answer_text(completion),
            self._entries_cache[answer_idx],
        )


def load_environment(
    num_train_examples: int = 10000,
    num_eval_examples: int = 500,
    seed: int = 42,
    dataset_path: str | None = None,
) -> vf.SingleTurnEnv:
    if dataset_path is None:
        raise ValueError("rg_mix requires a saved dataset_path (build one with build_dataset.py)")
    return RGMixEnv(
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        dataset_path=dataset_path,
    )
