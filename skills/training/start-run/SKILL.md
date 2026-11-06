---
name: start-run
description: How to launch prime-rl training runs — the `rl`, `sft`, and `inference` entrypoints, their config classes, and single-node/SLURM/dry-run modes. Use when starting a run or picking the right entrypoint.
---

# Start a run

All entrypoints run via `uv run <command>` and accept TOML configs via `@ path/to.toml` plus CLI overrides.

## Config system at a glance

[`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — Pydantic-based TOML + CLI loader. Highlights (see the `configs` skill for full mechanics):

- Config files via `@ path` (TOML / YAML / JSON); CLI args layer on top, deep-merged with class defaults.
- Nested groups via dotted CLI paths — kebab-case on the CLI, snake_case in TOML.
- Bool toggles: bare `--flag` enables, `--no-flag` disables (nested too).
- Lists: space-separated or JSON literal. Dicts: JSON literal, deep-merged with file values.
- Optional sub-configs (`WandbConfig | None`): bare `--wandb` enables defaults; `--wandb @ wandb.toml` enables from a file; `--no-wandb` disables.
- Discriminated unions are switched by the `type` tag (e.g. `--optimizer.type muon`).
- Validation aliases let renamed fields keep working; legacy keys can be remapped in a `model_validator(mode="before")`.
- Auto-generated `--help` panels from `Field(description=...)` or PEP 224 docstrings.
- Friendly errors: required-field boxes, validator errors point at the offending flag, unknown flags get a "did you mean" hint.

## `rl` — RL training

Launches inference server, orchestrator, and trainer as subprocesses. This branch
currently supports the native RG/reverse-text producer profile; historical legacy
examples require revision `1801d112` and its lock. Start from an explicitly
validated native config, not an automatic alias of an old `id`/`args` config.
Replace the path below with that config.

```bash
uv run rl @ /path/to/native-rl.toml
uv run rl @ /path/to/native-rl.toml @ /path/to/native-slurm.toml   # SLURM
uv run rl @ /path/to/native-rl.toml --dry-run                                # write scripts, don't run
```

- Config: `RLConfig` (`packages/prime-rl-configs/src/prime_rl/configs/rl.py`)
- Entrypoint: `src/prime_rl/entrypoints/rl.py`
- SLURM: single- and multi-node
- Validate launcher serialization, not only `RLConfig` JSON roundtrips: `write_subconfigs` omits `None` fields in TOML. Default-enabled optional sections such as model compilation, activation checkpointing and activation offloading therefore restore their defaults when child CLIs reload the file. Reload each generated component config through `cli(ComponentConfig, args=["@", path])` and compare with the intended resolved config before claiming these features are disabled. The safety pipeline harness records and checks both generated and actual child configs.
- Native tasksets use `env.taskset = { id = "reverse-text" }` or `id = "rg-mix-v1"` inside that table. Top-level environment `id`, `args` and `extra_env_kwargs` reject during configuration validation; the removed legacy bridge is not available in this profile. Keep the historical branch/lock for those runs and track environment migration before a main-branch replacement.
- Use `env.agent.harness = { id = "null" }` and `env.agent.runtime = { type = "subprocess" }`. Runtime now belongs to the agent. PrimeRL preserves subprocess as its default; an explicitly selected runtime remains honored. The old nested harness.runtime placement is migrated once, and conflicting placements reject. Do not silently provision the upstream default remote runtime.
- PrimeRL's environment wrapper composes native `EnvConfig` with serving settings from `ServeConfig`. `pool`, `address`, and `max_concurrent` remain on the PrimeRL environment entry; `env.taskset` and `env.agent` describe execution. Standalone `env-server` consumes the same resolved wrapper. Verify the generated child TOML, including these fields, before launch.
- Preflight the actual two-worker native server: the orchestrator loads tasks once and workers reconstruct the selected data. Exact task classes/configs and lossless task-data roundtrips are required. Explicit integer idx values, including duplicate/gappy values, are preserved; an unset idx gets its row position. Taskset views are not silently applied by the existing sampler. Infinite eval requires `num_examples`; training pulls remain bounded by admitted groups. Source RNG/cursor/generator state is not checkpointed, so resume remains a cold task restart.
- Await the official `EnvClient.run` final assembled Episode; streamed deltas are not completed rollouts. This slice accepts exactly one original trace before eligibility filtering, so multi-agent episodes reject even if only one trace is trainable. Failed or non-trainable traces cannot become credit. Generic plain shared-prefix branches remain supported; semantic/compaction branches reject before credit.
- The target-native profile has no accepted Harbor, multimodal, namespaced-tool or sampling-mask replay migration yet. Historical local Harbor admission checks do not establish compatibility with this dependency pair. Do not make an unavailable env work by blindly adding its package to the env extra or reusing an old source pin; review its TaskData/config/scoring APIs first.
- Engine weight-version spans are admitted only with validated sampled-token geometry. Raw opaque labels remain in the trace; comparable numeric live-policy IDs may travel on the physical token axis to the trainer as observation-only metadata. Missing/default/opaque labels stay unknown and never inherit the dispatch version. Sampling-mask replay remains unsupported; keep `return_sampling_mask` disabled. Startup remains `default`/unknown until actual full-weight adoption publishes a step. LoRA responses suppress unvalidated engine-global labels. Keep-mode updates retain in-flight KV/context, so sampled labels do not identify the weights that computed all context. See `experiments/weight-versions/README.md` and `experiments/backend-sync/README.md` for scope and actual acceptance receipts.
- The vLLM 0.32 backend profile retains `processed_logprobs`, dense V1 runner selection, precision defaults and value objectives. FA4 rejects before initialization; choose FA2/FA3. P/D plus routed-expert output is not supported by this profile. Flat `model.rope_scaling` dict/JSON uses a native HF override preserving checkpoint theta/partial rotation; nested schemes reject. Full-weight update failures leave the caller-paused engines paused, and the update itself is never automatically retried because NCCL adoption is not idempotent.
- Keep `VLLM_SERVER_DEV_MODE=0`: the native dev update protocol conflicts with PrimeRL's admin endpoints and startup rejects it. Read-only `/weight_info` and `/is_paused` are available without dev mode; do not enable mutation-oriented dev routes merely to inspect version or paused state.
- Renderer clients now use the native elastic multiplexed pool. Explicit `pool_size` rejects because fixed pool size has no equivalent multiplex setting; the former internal pool_size=1 is no longer emitted. This changes resource scheduling, not sampling settings, and carries no throughput claim.
- Monitor nested `logs/envs/train/<env>.log` and eval counterparts, not only the top-level orchestrator log. Use `experiments/native-producer/README.md` for exact current scope and evidence.
- Native RG uses `rg-mix-v1` with the same null/subprocess harness. It requires
  a saved dataset directory containing `metadata.json` and `dataset/`; see
  `packages/rg-mix-env/README.md`. Pin the dataset artifact and scorer version.
  Set the training slice length and evaluation slice explicitly; evaluation
  uses `split = "eval"` and begins immediately after the training slice. The
  orchestrator loads the cached data; workers reconstruct selected private
  scoring entries without reading that directory. These entries must never
  become policy messages or value prefixes. The accepted `seed` field does
  not regenerate or verify a cached dataset. Inline custom system text belongs in
  `env.taskset.prompt_text`; canonical `system_prompt` is now a file path, not
  inline text. Migrate old inline values explicitly and validate the file when
  using a file override.
- To retain a single-turn legacy run's post-generation training-token cap when
  moving it to a native taskset, set `training_projection = { type = "single_turn" }`
  on its `[[orchestrator.train.env]]` entry. The cap resolves to the actor's
  `seq_len`; `max_seq_len` can set an explicit positive cap. Full generated text,
  reward and provider usage are retained. All token-based training consumers
  (including filters and the critic) see the same owned prefix. The original
  physical trace remains accessible separately; serialized training traces
  carry a projection receipt. This option is disabled by default and is not an
  evaluation or generation limit. It rejects multi-turn, branched, tool,
  multimodal and unsupported token-sidecar traces rather than guessing their clipping
  semantics. Preflight the actual environment shape before a model run.
  The historical `experiments/sequence-boundary/README.md` records the parent-stack
  legacy comparison; its old harness/config IDs are not current launch recipes.

### Optional async value plane

An RL config with `[value_function]` always adds a `value-trainer`, which pulls
finalized rollouts from a bounded, nonblocking producer FIFO into a
rollout-granular replay buffer. Dedicated evaluation is the default and adds a
`value-evaluator` serving copy. Setting
`value_function.evaluator.placement = "trainer"` instead queues evaluation on
the live value trainer between updates. Do not add value loss or value-model
state to the policy trainer.

The evaluator's `/version` endpoint returns MessagePack `ValueVersionResponse`,
not JSON. Use `ValueEvaluatorClient` or decode the response bytes with
`msgspec.msgpack.decode(payload, type=ValueVersionResponse)` when probing it.

- Task-context rollouts are evaluated on arrival through the dynamic batcher.
  Sibling value versions may differ; group completion does not re-evaluate
  them. Leave-one-out peer context waits for the group for its first evaluation.
- Local runs reserve `deployment.num_value_train_gpus` after the policy GPUs.
  Dedicated placement also reserves `num_value_eval_gpus`; trainer placement
  requires it to resolve to zero.
- Multi-node runs similarly reserve value-trainer nodes and reserve evaluator
  nodes only in dedicated placement. Start from
  the deployment/value sections of `examples/value_function/rl.toml`, explicitly
  replacing its legacy environment entry with a validated native configuration;
  set scheduler fields for the target environment.
- Privileged value context defaults to an optional static native-v1 task
  `value_function_prompt` declared by the task's own data type (RG declares it;
  the upstream TaskData base does not). Confirm it survives worker reconstruction
  and is redacted in public records. Set `value_function.privileged_context =
  "group_leave_one_out"` to use each complete GRPO group member's K-1 peer
  trajectories and rewards instead. Conditioned branches are never truncated
  and must fit `value_function.model.seq_len`.
- GRPO value-backed baseline choices are `value` and `tether`. TETHER is
  adaptive by default, uses a leave-one-out anchor, inherits its regression
  window from `value_function.batch_size`, initializes `rho` at zero, and uses
  EMA decay `0.95`. It subtracts `B + rho * (V-B)` from the fixed critic policy
  lambda-return, starting at `Q_lambda - B_LOO` (also exact LOO when policy
  lambda is one). Set `baseline.adaptive = "None"` to use the static `rho = 0.5`
  coefficient. To opt into branch-local fixed action buckets, configure
  `baseline.adaptive.position.bin_size`; sparse buckets hold their previous
  coefficient, and the global one-coefficient mode remains the default.
- Compare `algorithm/<env>/tether/mse_applied` with `mse_group_applied` for
  the latest scored group's applied coefficients versus the group anchor.
  `mse_batch_fit` and `mse_post_fit_ema` are same-window fitting diagnostics.
- Branched envs need no declaration at the all-ones return defaults. If their
  branches are sequential episode segments and discounted or lambda-based
  credit must cross them, set the per-env algorithm's
  `branch_semantics = "sequential"`; do not use it for parallel forks.
- Launch trainer placement through `rl`, not the standalone `value-trainer`
  command; the managed run-done file owns serve-only shutdown.
- Check `logs/value_trainer.log`, evaluator `/health` and `/version`, plus
  `logs/value_evaluator.log` in dedicated placement and
  producer-queue and `value/replay_*` metrics. `value_function.batch_size`
  inherits the policy rollout batch size unless set explicitly; token-batched
  policy runs must set it explicitly. The producer retains 2048 pending
  rollouts by default and drops the oldest on overload; value work must not
  backpressure policy rollout generation.
- Value checkpoints live under `<output_dir>/value` and have an independent
  version. `warmup_updates` gates only policy-batch shipping while generation
  and value training continue. The gate uses the minimum evaluator version in
  the exact post-filter cohort, so stale cohorts remain blocked after the live
  evaluator advances. Batches with no value-scored samples use the live
  evaluator version, preserving an explicitly configured global barrier.
  Replay uses the same rules during and after warmup.
  `replay.max_updates_per_rollout` is a hard per-rollout selection cap;
  its default capacity and refill threshold are both
  `max_updates_per_rollout * value_function.batch_size`. Replay state is not
  checkpointed and refills from fresh rollouts after resume.
## `sft` — SFT training

Launches torchrun internally — never call torchrun directly.

```bash
uv run sft @ examples/reverse_text/sft.toml
uv run sft @ examples/reverse_text/sft.toml --slurm
uv run sft @ examples/reverse_text/sft.toml --dry-run
```

- Config: `SFTConfig` (`packages/prime-rl-configs/src/prime_rl/configs/sft.py`)
- Entrypoint: `src/prime_rl/entrypoints/sft.py`
- SLURM: single- and multi-node

## `inference` — vLLM server

OpenAI-compatible API plus prime-rl custom endpoints (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster`). Always use this entrypoint — never `vllm serve` directly.

```bash
uv run inference @ configs/debug/infer.toml
uv run inference --model.name Qwen/Qwen3-0.6B --model.enforce-eager
```

Smoke checks:

```bash
curl http://<host>:<port>/health
curl http://<host>:<port>/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

- Config: `InferenceConfig` (`packages/prime-rl-configs/src/prime_rl/configs/inference.py`)
- Entrypoint: `src/prime_rl/entrypoints/inference.py`
- SLURM: single-node, multi-node, and disaggregated deployments

## Summary

| Command | Purpose | Typical use |
|---------|---------|-------------|
| `rl` | Full RL pipeline, optionally including the value plane | Production RL training |
| `sft` | Supervised fine-tuning | SFT and hard-distill |
| `inference` | vLLM server | Standalone serving / debugging |
| `value-trainer` | Standalone FSDP critic trainer | Debugging a resolved `value.toml` |
| `value-evaluator` | Standalone dedicated critic serving replica | Debugging a resolved `value.toml` |

## Key paths

- `src/prime_rl/entrypoints/` — `rl`, `sft`, `inference` (+ `trainer`, `orchestrator`, `value-trainer`, and `value-evaluator` for direct launches)
- `packages/prime-rl-configs/src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs (e.g. `reverse_text/`)

## Value behavior checks

Before changing shared training infrastructure around the value plane, use
`experiments/value-baseline/README.md` for independent numerical comparisons,
source-verified regression checks, mutation sensitivity and real FSDP restart
checks. Run pytest in an isolated test pod: its cleanup fixture terminates
matching training processes. Keep test source, output directories and runtime
environments separate from active experiments. Use a fresh process for critic
checkpoint resume; runtime monitoring is process-global. Tiny-model checks do
not replace model-specific or live evaluator/transport validation.

Renderer parity is limited to the tested default single-turn RG/reverse cases.
A supplemental disabled-thinking multi-turn fixture intentionally changes its
physical prompt from47 to51 tokens under the new Qwen renderer, retaining the
historical empty think wrapper; the generation suffix matches. This upstream
prompt change can affect continuations even though private value ownership and
credit math are unchanged. Do not claim universal prompt parity.

The launcher enables vLLM scale-out HTTP routes for the required `/inference/v1/generate` token endpoint. A direct native vLLM launch needs `--enable-scale-out`; otherwise this endpoint returns403 even when health and model discovery work. This flag registers HTTP routes and does not itself enable P/D or KV transfer.

For finite NCCL runs, the orchestrator stops scheduling training after the final successful shipment, finishes triggered evaluations, and keeps the receiver alive through the last policy broadcast the actor promises. Do not replace a stalled final run with an extra synthetic batch or broadcast. On NCCL resume, both actor and orchestrator must restart: the actor transfers its restored checkpoint before requesting new batches, and the orchestrator completes that transfer before opening rollout sources. Check actual adopted policy and restored critic versions in the logs; unused tail rollouts can change the final critic update count. Filesystem resume continues to use checkpoint artifacts.

Sampled policy diagnostics use the existing RL/reference-KL token population. `observed_policy_version` reports coverage; `sampled_weight_age` compares known nonfuture labels against the pre-update policy version only for single-run, non-LoRA training. Unknown, future and empty populations remain explicit. These observations do not identify critic versions or the full retained KV history; see `experiments/provenance-diagnostics/README.md`.

Use `mismatch_kl_by_sampled_age` to inspect whether mismatch is concentrated in older sampled tokens. Compare bin counts and saturation fractions alongside means/maxima; unknown/future are separate populations and absent age bins are not evidence of fresh data. This diagnostic is observation-only and does not enable stale-token masking or adaptive concurrency.
