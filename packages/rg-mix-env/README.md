# RG mix environment

This native producer profile provides the `rg-mix-v1` taskset, using the same five reasoning-gym variants and scorer dependency pinned to `reasoning-gym==0.1.25`. Historical `rg-mix-env` source is retained, but requires the frozen current-stack branch and lock; it cannot execute under this profile’s native-only verifier.

The native taskset reads an existing generated dataset directory containing `metadata.json` and `dataset/`. The orchestrator selects task data; workers receive the selected question and private scoring entry and create only a lightweight scorer. Workers do not load or regenerate the dataset.

```toml
[[orchestrator.train.env]]
env.taskset = { id = "rg-mix-v1", dataset_path = "/data/rg-dataset", num_train_examples = 10000, num_eval_examples = 256, seed = 42, split = "train" }
env.agent.harness = { id = "null" }
env.agent.runtime = { type = "subprocess" }
```

For evaluation, configure the same taskset with `split = "eval"`. Evaluation takes rows immediately following the configured training slice. Task indices start at zero within each split; the private scorer index remains the global dataset index. The `seed` field is accepted for configuration compatibility and is unused when selecting cached rows. Dataset contents and their recorded hashes establish provenance; this field is not checked against the artifact metadata.

The policy receives the original reasoning-gym system prompt and question. Private scoring data never becomes a policy message or value-function prefix. The task takes one model turn and scores its answer with the selected reasoning-gym scorer. Answer extraction follows the package's XML answer convention; the format metric has weight zero and does not affect reward. There is no group-dependent reward function.

To create a new public artifact using the installed, pinned scorer dependency, run from the repository root:

```bash
uv run python packages/rg-mix-env/generate_rg_mix_dataset.py \
  --output ./data/rg-dataset --num-train 10000 --num-test 256 --seed 42 --workers 4
```

Then edit `dataset_path` in your native RG config to the generated directory. Generation is explicit and may be expensive; loading the native taskset never regenerates questions. Save the generator arguments, dependency lock and artifact hashes with each experiment. This creates a new artifact; it does not reproduce an unavailable private cache by assertion.

The generator imports shared task/scorer definitions directly and runs with the current native verifier. The retained legacy environment class requires its separately frozen runtime.

For a native training configuration that must retain the legacy actor-length
cap, add `training_projection = { type = "single_turn" }` beside the `env`
settings on the training entry. It resolves to the actor sequence length and
clips only the owned training trace after full-response scoring. The generated
text and reward stay complete; policy filters and value inputs use the same
retained tokens. Omit this option to retain full native training geometry.
This projection is not a generation or evaluation limit.

The RG default system prompt is unchanged. To provide inline text, use
`env.taskset.prompt_text`; the canonical `env.taskset.system_prompt` now names a
UTF-8 file whose contents override the task prompt. Migrate an older inline
`system_prompt` value to `prompt_text` explicitly; it must not be interpreted as
a filesystem path. `RGMixData.value_function_prompt` is an optional custom
critic-only prefix; the scorer entry is never used as that prefix.
