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

- Local runs reserve `deployment.num_value_train_gpus` after the policy GPUs.
  Dedicated placement also reserves `num_value_eval_gpus`; trainer placement
  requires it to resolve to zero.
- Multi-node runs similarly reserve value-trainer nodes and reserve evaluator
  nodes only in dedicated placement. Use `examples/value_function/rl.toml` as
  the local smoke and `configs/rg_mix/async_value.toml` as the dedicated
  four-node shape.
- Launch trainer placement through `rl`, not the standalone `value-trainer`
  command; the managed run-done file owns serve-only shutdown.
- Before a local GPU smoke on Perlmutter, point compiler and package caches at
  scratch (`UV_CACHE_DIR`, `XDG_CACHE_HOME`, `HF_HOME`, `TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH`, and `TMPDIR`). Triton otherwise
  defaults to the quota-limited home filesystem and can fail during vLLM startup.
  Keep `TMPDIR` short (for example, `$SCRATCH/t/pv`) because vLLM IPC socket
  paths must also fit the Unix-domain socket path limit.
- Check `logs/value_trainer.log`, evaluator `/health` and `/version`, plus
  `logs/value_evaluator.log` in dedicated placement and
  producer-queue and `value/replay_*` metrics. `value_function.batch_size`
  inherits the policy rollout batch size unless set explicitly; token-batched
  policy runs must set it explicitly. The producer retains 2048 pending
  rollouts by default and drops the oldest on overload; value work must not
  backpressure policy rollout generation.
- Value checkpoints live under `<output_dir>/value` and have an independent
  version. `warmup_updates` gates only policy-batch shipping while generation
  and value training continue. Replay uses the same rules during and after
  warmup. `replay.max_updates_per_rollout` is a hard per-rollout selection cap;
  its default capacity and refill threshold are both
  `max_updates_per_rollout * value_function.batch_size`. Replay state is not
  checkpointed and refills from fresh rollouts after resume.
- TETHER can fit its two coefficients online by adding an empty
  `[orchestrator.algo.baseline.adaptive]` table. Adaptive mode requires the
  leave-one-out anchor, starts from alpha/rho zero, inherits the critic rollout
  batch size, and updates during value warmup. Watch
  `algorithm/<env>/tether/{alpha,rho,updates,mse_ema}`; orchestrator checkpoints
  preserve the estimator state.
- TETHER position conditioning is opt-in through
  `[orchestrator.algo.baseline.position]`. It uses fixed-width, branch-local
  generated-action bins (`bin_size = 1024` by default). With adaptive TETHER,
  watch the per-bin coefficient/support metrics under
  `algorithm/<env>/tether/position/bin_NNN/*` as well as the active-bin summary.

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
