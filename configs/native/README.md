# Native training profile

These are current, authored launch examples for the native reverse-text and cached RG producers. Each runs at most three policy updates, uses groups of eight, and writes W&B logs offline. Policy-only examples request two GPUs; value examples request four (actor, inference, critic trainer, dedicated critic evaluator). Defaults for precision, activation checkpointing, compilation and value loss are retained.

| Environment | Policy only | Independent value function |
| --- | --- | --- |
| Reverse text | [reverse_text_policy.toml](reverse_text_policy.toml) | [reverse_text_value.toml](reverse_text_value.toml) |
| Cached RG mix | [rg_mix_policy.toml](rg_mix_policy.toml) | [rg_mix_value.toml](rg_mix_value.toml) |

From the repository root, inspect the resolved children before allocating resources:

```bash
uv run rl @ configs/native/reverse_text_value.toml --dry-run
```

Then launch on a suitably provisioned machine:

```bash
WANDB_MODE=offline uv run rl @ configs/native/reverse_text_value.toml
```

Use a fresh output directory for each run (`--output-dir outputs/my-run`). The configs set process-local W&B offline mode; the environment variable also prevents shared-run initialization from attempting an online connection. Explicitly enable online tracking only when intended.

The examples name the public Qwen3-4B model and explicitly select its renderer. The earlier backend acceptance used model revision `cdbee75f17c01a7cc42f958dc650907174af0554`. To reuse those exact weights, resolve that revision to a local snapshot and pass `--model.name /path/to/snapshot`; the public model name alone is not an immutable weight pin.

RG requires an existing `/data/rg-dataset` artifact with `metadata.json` and `dataset/`, mounted where the orchestrator runs. The taskset does not generate data; the public [dataset generator](../../packages/rg-mix-env/README.md) creates the required artifact explicitly. No private cache access is required. Edit `dataset_path` in the RG TOML for your mount. Indexed CLI overrides into the environment list are not supported by the current config loader. Training selects the first 10,000 cached rows; evaluation selection, when configured separately, starts after those rows and is capped at 256. These examples do not launch evaluation. Task identities are split-local, while scoring answers retain the original global cache indices. See the [RG package](../../packages/rg-mix-env/README.md) for artifact provenance and scorer requirements.

RG has a 2,048-token actor budget and 128 output tokens; reverse-text uses 512/64. Inspect your cached prompt lengths before running: the RG budget is a smoke configuration, not a claim that all tasks fit. Native physical traces are retained by default. The explicit single-turn legacy training projection is described in [sequence-boundary acceptance](../../experiments/sequence-boundary/README.md); it is not enabled here.

Value examples use a two-update critic warmup, critic batches of eight and one replay update per rollout. The actor uses the pure value baseline; the policy-only configs explicitly select the mean baseline. Both retain ancestral sampling (`temperature=top_p=1`, no top-k/min-p filtering or penalties).

CPU acceptance checks the actual registry/factories, all four role layouts, native/value options, and the launcher's TOML-to-child-CLI roundtrip. Earlier [backend lifecycle acceptance](../../experiments/backend-sync/README.md) exercised Qwen4B fresh/resume with a smaller group-two fixture. These new group-eight examples have not had a separate GPU run; do not treat configuration acceptance as a throughput or convergence result.

The [compatibility inventory](../../experiments/dependency-migration/config-compatibility.json) classifies every historical config and its intended entrypoint. Historical legacy environments require the frozen [source and lock at `1801d112`](https://github.com/HyperPotatoNeo/prime-values/tree/1801d112); the earlier [`8474e835` milestone](https://github.com/HyperPotatoNeo/prime-values/tree/8474e835) retains the older backend stack. Historical files are preserved, not silently remapped to native tasksets. The selected migration scope supports RG mix and reverse-text while retaining that historical runtime separately. The full historical environment/model/MM/Harbor/P-D catalog remains outside this profile.
