# rg-mix-env

Weighted mix of challenging reasoning-gym tasks for RL training.

## Overview

This environment combines 5 challenging reasoning tasks, weighted **inversely proportional** to their Qwen3-4B pass@1 scores so that harder tasks get more representation during training.

## Tasks

| Task | Type | pass@1 | Weight (1/pass@1) | Config |
|------|------|--------|-------------------|--------|
| arc_1d | Pattern recognition | 0.40 | 2.49 | default |
| sokoban_hard | Planning/search | 0.31 | 3.23 | 3-4 boxes, 9x9 |
| countdown_7 | Arithmetic search | 0.30 | 3.33 | 7 numbers |
| zebra_puzzles_7 | Constraint satisfaction | 0.25 | 3.98 | 7 people, 5 chars |
| cryptarithm | Cryptarithmetic | 0.19 | 5.31 | default |

## Quickstart

```python
from verifiers import load_environment

env = load_environment("rg-mix-env")

# or with custom args:
env = load_environment("rg-mix-env", num_train_examples=10000, num_eval_examples=2048, seed=42)
```

## Environment Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `num_train_examples` | int | 10000 | Number of training examples |
| `num_eval_examples` | int | 2048 | Number of evaluation examples |
| `seed` | int | 42 | Random seed for reproducibility |
| `dataset_path` | str \| None | None | Path to pre-generated dataset directory. If provided, loads from disk (~10s) instead of generating (~23 min). |

## Pre-generating Datasets

Dataset generation is slow (~23 min) due to puzzle verification (zebra puzzles, sokoban BFS). Pre-generate once and reuse:

```bash
# Generate dataset (uses multiprocessing, ~11 min with 32 workers)
python generate_rg_mix_dataset.py \
    --output /pscratch/sd/s/siddart2/datasets/rg_mix_7500 \
    --num-train 7500 --num-test 100 --seed 42

# Or submit as batch job
sbatch generate_dataset.sbatch
```

Then reference in TOML config:

```toml
[[orchestrator.env]]
id = "rg-mix-env"
args = { num_train_examples = 7500, num_eval_examples = 100, seed = 42, dataset_path = "/pscratch/sd/s/siddart2/datasets/rg_mix_7500" }
```

### What gets saved

The `--output` directory contains:
- `dataset/` — HF Dataset with `question`, `answer`, `task` columns
- `metadata.json` — Entry map + full entry dicts for scoring (~9 MB for 7600 examples)

### Pre-generated datasets on cluster

| Path | Train | Test | Seed |
|------|-------|------|------|
| `$SCRATCH/datasets/rg_mix_7500` | 7500 | 100 | 42 |

## Requirements

Requires `verifiers[rg]` and `reasoning-gym` to be installed in the runtime environment.
