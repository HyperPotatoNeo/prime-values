# RG mix environment

The package provides the `rg-mix-env` legacy environment and the `rg-mix-v1` canonical native taskset. Both use the same five reasoning-gym variants and deterministic scorer dependency, pinned to `reasoning-gym==0.1.25`.

The native taskset reads an existing generated dataset directory containing `metadata.json` and `dataset/`. The orchestrator selects task data; workers receive the selected question and private scoring entry and create only a lightweight scorer. Workers do not load or regenerate the dataset.

```toml
[[orchestrator.train.env]]
env.taskset = { id = "rg-mix-v1", dataset_path = "/data/rg-dataset", num_train_examples = 10000, num_eval_examples = 256, seed = 42, split = "train" }
env.agent.harness = { id = "null", runtime = { type = "subprocess" } }
```

For evaluation, configure the same taskset with `split = "eval"`. Evaluation takes rows immediately following the configured training slice. Task indices start at zero within each split; the private scorer index remains the global dataset index. The `seed` field is accepted for configuration compatibility and is unused when selecting cached rows. Dataset contents and their recorded hashes establish provenance; this field is not checked against the artifact metadata.

The policy receives the original reasoning-gym system prompt and question. Private scoring data never becomes a policy message or value-function prefix. The task takes one model turn and scores its answer with the selected reasoning-gym scorer. Answer extraction follows the package's XML answer convention; the format metric has weight zero and does not affect reward. There is no group-dependent reward function.

`rg_mix_env.load_environment` supports the existing legacy dataset-loading and generation APIs. `generate_rg_mix_dataset.py` creates dataset artifacts for either entrypoint. The native taskset requires a cached artifact so its selected private inputs survive worker reconstruction without regenerating questions.
