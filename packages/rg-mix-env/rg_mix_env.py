"""Legacy RG mix environment, with its dataset/scorer definitions shared with the native taskset."""

from typing import Optional

import verifiers as vf
from datasets import Dataset
from rg_mix_shared import DEFAULT_SYSTEM_PROMPT, _generate_dataset, _load_dataset
from rg_mix_shared import TASK_VARIANTS as TASK_VARIANTS


class RGMixEnv(vf.SingleTurnEnv):
    """Weighted mix of multiple reasoning-gym task variants.

    Unlike ReasoningGymEnv which uses CompositeDataset (no duplicate names),
    this env creates separate datasets per variant and builds the HF dataset
    with proportional sampling.
    """

    def __init__(
        self,
        num_train_examples: int = 10000,
        num_eval_examples: int = 2048,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        parser: Optional[vf.Parser] = None,
        seed: int = 42,
        dataset_path: Optional[str] = None,
    ):
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.seed = seed

        if dataset_path is not None:
            rows, self._entry_map, self._entries_cache, self._variant_datasets = _load_dataset(dataset_path)
        else:
            rows, self._entry_map, self._entries_cache, self._variant_datasets = _generate_dataset(
                num_train_examples, num_eval_examples, seed
            )

        # Task is a structured payload in current Verifiers; keep the variant
        # label as metadata instead of the deprecated string task route.
        rows = [
            {"question": row["question"], "answer": row["answer"], "info": {"variant_id": row["task"]}} for row in rows
        ]

        # Split into train/eval
        train_rows = rows[:num_train_examples]
        eval_rows = rows[num_train_examples : num_train_examples + num_eval_examples]
        dataset = Dataset.from_list(train_rows)
        eval_dataset = Dataset.from_list(eval_rows)

        # Set up parser and rubric
        parser = parser or vf.XMLParser(fields=["answer"])
        rubric = vf.Rubric(parser=parser)

        env_ref = self

        async def check_answer_reward_func(completion: vf.Messages, answer: str, **kwargs) -> float:
            global_idx = int(answer)
            vid, entry_idx = env_ref._entry_map[global_idx]
            ds = env_ref._variant_datasets[vid]
            entry = env_ref._entries_cache[global_idx]
            response = str(parser.parse_answer(completion)).strip()
            reward = ds.score_answer(answer=response, entry=entry)
            return reward

        rubric.add_reward_func(check_answer_reward_func)
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.0)

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            message_type="chat",
        )
        self.parser = parser
        self.rubric = rubric

    def score_candidate(self, answer_idx: int, response: str) -> float:
        """Score a candidate response for a given problem index."""
        vid, entry_idx = self._entry_map[answer_idx]
        ds = self._variant_datasets[vid]
        entry = self._entries_cache[answer_idx]
        return ds.score_answer(answer=response, entry=entry)


def load_environment(
    num_train_examples: int = 10000,
    num_eval_examples: int = 2048,
    seed: int = 42,
    dataset_path: Optional[str] = None,
) -> vf.Environment:
    """Load the weighted reasoning-gym mix environment.

    Args:
        dataset_path: Optional path to a pre-generated dataset directory.
            If provided, loads from disk instead of generating (much faster).
    """
    return RGMixEnv(
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        dataset_path=dataset_path,
    )
