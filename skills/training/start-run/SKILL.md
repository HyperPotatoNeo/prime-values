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

Launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/reverse_text/rl.toml
uv run rl @ examples/reverse_text/rl.toml @ examples/reverse_text/slurm_rl.toml   # SLURM
uv run rl @ examples/reverse_text/rl.toml --dry-run                                # write scripts, don't run
```

- Config: `RLConfig` (`packages/prime-rl-configs/src/prime_rl/configs/rl.py`)
- Entrypoint: `src/prime_rl/entrypoints/rl.py`
- SLURM: single- and multi-node
- Validate launcher serialization, not only `RLConfig` JSON roundtrips: `write_subconfigs` omits `None` fields in TOML. Default-enabled optional sections such as model compilation, activation checkpointing and activation offloading therefore restore their defaults when child CLIs reload the file. Reload each generated component config through `cli(ComponentConfig, args=["@", path])` and compare with the intended resolved config before claiming these features are disabled. The safety pipeline harness records and checks both generated and actual child configs.
- Native v1 tasksets use `taskset = { id = "reverse-text-v1" }`, not a top-level `id` (which selects the legacy `load_environment` interface). For reverse text, also set `harness = { id = "null", runtime = { type = "subprocess" } }`; see `configs/debug/v1/reverse_text.toml`. Preflight the actual native `EnvServer` constructor and dataset loading before launching model processes. Monitor `logs/` recursively: environment failures appear in nested `envs/train/<env>.log`, not necessarily the top-level orchestrator log.
- Environment packages: before launching a config with a non-core verifier env id,
  verify the package imports under `uv run` (for example
  `uv run python -c "import importlib.util; print(importlib.util.find_spec('r2e_gym_v1'))"`).
  If a local env exists under `deps/research-environments/environments/` but does not
  import, add it to the root `pyproject.toml` env extra, workspace members, and
  `[tool.uv.sources]`, then run `uv sync --all-extras`.

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
  `examples/value_function/rl.toml` and set the multi-node deployment and
  scheduler fields for the target environment.
- Privileged value context defaults to an optional static native-v1 task
  `value_function_prompt`. Set `value_function.privileged_context =
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

## Score-centering / EMA research runs

Use `experiments/score-centering-ema/RESEARCH.md` for the matched comparison.
Apply the recorded source patches to the exact pinned renderer/verifier submodules
with `uv run --no-project python experiments/score-centering-ema/apply_dependency_patches.py`
before syncing dependencies. Never patch an installed virtual environment.

`prepare.py` only emits configurations; Kubernetes `suspend: true` does not prevent
Kueue from admitting an applied job. Validate component CLI reloads and use server
dry-run before submission. The external four-role launcher uses headless services,
distinct node ranks, filesystem broadcasts and shared completion/failure flags.
Use fresh run names/output paths, and securely provision the referenced W&B Secret
at launch. Check live capacity and account for other jobs before choosing concurrency.

Score centering requires the actual sampler's full-normalized top-k probabilities
through native and legacy token paths. Keep generation ancestral. It is supported
by the research REINFORCE/TIS loss, custom Qwen3-4B, CP=1 and the chunked output head;
do not silently combine it with DPPO/PPO clipping. The injected model forward must
route candidate IDs to the output head, never the transformer backbone.

EMA uses a frozen sharded model on trainer GPUs, updates once per publication, and
checkpoints its state and clock with the optimizer. Verify fresh-process resume.
Retain zero-advantage samples consistently across arms so EMA KL also applies to
them. With asynchronous collection, use acknowledged filesystem cleanup and an
explicit canonical checkpoint weight directory; keep durable checkpoints separate
from ephemeral broadcasts. Enable bounded consumed-rollout binary retention for
long top-k runs; older trajectory JSON is retained losslessly compressed. Check
disk capacity before every wave, including retained checkpoints from finished runs.
Dispatcher lead is a scheduling threshold, not an exact
policy-age guarantee: already-in-flight requests can finish after the gate closes.

Functional smoke tests may be offline; substantive runs use online W&B. Inspect
trajectories, loss/gradient finiteness, mismatch tails, actual policy age, errors and
truncation. The legacy in-loop evaluation is not held-out; explicitly select
`get_eval_dataset()` for the frozen validation/test split. Never run pytest beside
an active smoke: the test cleanup fixture terminates matching training processes.

For the authorized score-centering/EMA sweep, use the generated campaign `lane*.yaml` manifests: each four-node lane runs a fixed sequential list with filesystem barriers and failure propagation. Applying all individual manifests would exceed the intended concurrency. Regenerate and run `preflight.py` after parameter changes. A lane refuses an existing run output directory; recovery needs an explicitly planned fresh attempt/resume, not an automatic restart. Local scalar JSONL logging is enabled by `PRIME_RL_METRICS_FILE`.

The November6 score-centering campaign was expanded after its initial launch: consult `prepared/siddarth-sc-ema-20261106/expanded/schedule.json`. Seven runs moved to independent Jobs while the original three continue unchanged. Reserved output directories make the original lane no-overwrite guards retire them after their current experiment, producing an expected lane-level Failed status; distinguish this from an experiment failure and check per-run policy.done/metrics. Never reapply the original sequential schedule or clear reassignment guards.

Final user constraint for this campaign is six concurrent runs (192 GPUs). `queue_campaign.py` manages four remaining runs from the Mac using the official helper; preserve `results/queue-192/state.json` on restart, check process and last_success, and never run two supervisors concurrently. API failures fail closed. No namespace RBAC modification is available; do not imply the local queue survives Mac sleep/shutdown or unrefreshable auth.

Performance checks for this deployment: NFS negative directory caching can hide
new per-step batch files for about a minute across trainer nodes. Readiness polling
uses `path_exists_fresh`, which refreshes parent directory entries only after a
missing-path result. Test on two nodes; do not remount shared storage globally.
For token-only nonstreaming inference, avoid unused candidate detokenization and
intermediate cumulative responses, while retaining string-stop and prompt-logprob
text behavior. Compact score-centering heads must keep actual full-normalized
sampler probabilities; never lower the engine's requested logprob count to save
response formatting work.

The performance relaunch uses `prepared-fast1/manifest.json`, fresh `-fast1` attempt
names, and `queue_campaign.py --manifest ... --max-runs 7` (224 GPUs), as authorized
by Siddarth. Original attempts remain archived. JSON component configs preserve
explicit null activation-offload settings; `preflight_fast.py` compares all 30
child CLI configs against the original algorithm configuration. Retain eight
shards per trainer replica and activation checkpointing; disable activation CPU
offloading. Keep output/config paths and runtime source forks separate from live
attempts. Count validation allocations before starting the seven-run queue.
