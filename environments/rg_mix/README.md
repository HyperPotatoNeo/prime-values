# rg-mix

Mixed reasoning-gym single-turn environment used for the value-corrected GRPO
baseline experiments. It combines five reasoning-gym task variants — `arc_1d`,
`sokoban_hard`, `countdown_7`, `zebra_puzzles_7`, `cryptarithm` — with rewards in
`[0, 1]` from each variant's `score_answer`. The model answers inside
`<answer>...</answer>` tags (parsed by `vf.XMLParser`).

Prompts are loaded from a **fixed saved dataset** so every run trains on the same
data. This mirrors the offline static-value diagnostics, so their coefficient
recommendations transfer online.

## Build the dataset (once)

```bash
uv run --extra rgmix python environments/rg_mix/build_dataset.py \
  --out /path/to/rg_mix_10k --total 10500 --seed 42
```

This writes `dataset/` (HF dataset of `{question, answer}` rows) and
`metadata.json` (`entry_map` + `entries_cache`) under `--out`.

## Use in a config

```toml
[[orchestrator.train.env]]
id = "rg-mix"
name = "rg-mix"
args = { dataset_path = "/path/to/rg_mix_10k", num_train_examples = 10000, num_eval_examples = 500, seed = 42 }
```

Install with `uv sync --extra rgmix` (installs this env + `reasoning-gym`).
