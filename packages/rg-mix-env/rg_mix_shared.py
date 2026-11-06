"""Dataset and scorer definitions shared by the RG mix entrypoints."""

import json
import random
from pathlib import Path

import reasoning_gym as rg
from datasets import load_from_disk
from reasoning_gym.utils import SYSTEM_PROMPTS

DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]

# Final candidate tasks — challenging tasks with inverse pass@1 weighting
# pass@1 scores from Qwen3-4B feasibility evals (temp=1.0, 8192 tokens)
TASK_VARIANTS = [
    {
        "id": "arc_1d",
        "task": "arc_1d",
        "pass_at_1": 0.4016,
        "config": {},
    },
    {
        "id": "sokoban_hard",
        "task": "sokoban",
        "pass_at_1": 0.3101,
        "config": {"min_boxes": 3, "max_boxes": 4, "max_w": 9, "max_h": 9},
    },
    {
        "id": "countdown_7",
        "task": "countdown",
        "pass_at_1": 0.30,  # estimated from trend: 4->0.89, 5->0.68, 6->0.47
        "config": {"min_numbers": 7, "max_numbers": 7},
    },
    {
        "id": "zebra_puzzles_7",
        "task": "zebra_puzzles",
        "pass_at_1": 0.2510,
        "config": {"num_people": 7, "num_characteristics": 5},
    },
    {
        "id": "cryptarithm",
        "task": "cryptarithm",
        "pass_at_1": 0.1882,
        "config": {},
    },
]

# Compute inverse weights (1/pass@1) — harder tasks get more samples
for v in TASK_VARIANTS:
    v["weight"] = 1.0 / v["pass_at_1"]


def _generate_dataset(num_train_examples, num_eval_examples, seed):
    """Generate the rg-mix dataset from scratch using reasoning-gym.

    Returns (rows, entry_map, entries_cache, variant_datasets) where:
    - rows: list of dicts with question/answer/task keys
    - entry_map: list of (variant_id, entry_idx) tuples for scoring
    - entries_cache: dict of global_idx -> entry dict for scoring
    - variant_datasets: dict of variant_id -> rg ProceduralDataset
    """
    total_examples = num_train_examples + num_eval_examples

    variant_datasets = {}
    for i, variant in enumerate(TASK_VARIANTS):
        ds = rg.create_dataset(
            variant["task"],
            seed=seed + i + 1,
            size=total_examples,
            **variant["config"],
        )
        variant_datasets[variant["id"]] = ds

    entry_map = []
    entries_cache = {}
    rng = random.Random(seed)
    weights = [v["weight"] for v in TASK_VARIANTS]

    rows = []
    for i in range(total_examples):
        chosen_idx = rng.choices(range(len(TASK_VARIANTS)), weights=weights, k=1)[0]
        variant = TASK_VARIANTS[chosen_idx]
        vid = variant["id"]
        ds = variant_datasets[vid]

        entry = ds[i % len(ds)]
        global_idx = len(entry_map)
        entry_map.append((vid, i % len(ds)))
        entries_cache[global_idx] = entry

        rows.append(
            {
                "question": entry["question"],
                "answer": str(global_idx),
                "task": vid,
            }
        )

    return rows, entry_map, entries_cache, variant_datasets


def _load_dataset(dataset_path):
    """Load a pre-generated rg-mix dataset from disk.

    Returns (rows, entry_map, entries_cache, variant_datasets).
    Variant datasets are created as lightweight instances (size=1) just for
    their score_answer method — no slow entry generation.
    """
    dataset_path = Path(dataset_path)

    with open(dataset_path / "metadata.json") as f:
        meta = json.load(f)

    entry_map = [tuple(x) for x in meta["entry_map"]]
    # entries_cache keys were stringified by JSON — convert back to int
    entries_cache = {int(k): v for k, v in meta["entries_cache"].items()}

    hf_ds = load_from_disk(str(dataset_path / "dataset"))
    rows = [dict(row) for row in hf_ds]

    # Create lightweight rg datasets (size=1) for scoring only
    variant_datasets = {}
    for variant in TASK_VARIANTS:
        vid = variant["id"]
        ds = rg.create_dataset(
            variant["task"],
            seed=1,
            size=1,
            **variant["config"],
        )
        variant_datasets[vid] = ds

    return rows, entry_map, entries_cache, variant_datasets
