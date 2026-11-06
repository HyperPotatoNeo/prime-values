# Overview

Prime Values extends `prime-rl`, a framework for large-scale, asynchronous reinforcement learning of large language models. It is designed to be easy to use and hackable, yet capable of training 1T+-parameter MoE models on 1000+ GPU clusters.

## Architecture

A `prime-rl` RL run is three cooperating processes:

![Architecture](assets/architecture.png)

- **Inference** — vLLM-backed server (or fleet) holding the current policy. The orchestrator drives rollouts through the token-in `/v1/generate` route via the [`renderers`](https://github.com/PrimeIntellect-ai/renderers) package (OpenAI-compatible chat/completions routes are also exposed for external clients). We are trying to stay up-to-date with the latest vLLM features, you can read more about the supported features and deployment options in the dedicated [inference documentation](inference.md).
- **Orchestrator** — Lightweight CPU process that owns the data plane across many [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) training and eval environments. Each env runs in an isolated subprocess with a variable-size pool of env workers for scalability. The orchestrator drives multi-turn rollouts against the inference fleet (tool use, browsers, sandboxes, long horizons) without re-tokenizing across turns, computes advantages, packs the rollouts into training batches, and relays new weights from trainer to inference.
- **Trainer** — FSDP2 process group that consumes packed rollouts and steps the optimizer. We ship optimized custom modeling code for many MoE / dense / VLM families that unlocks advanced trainer parallelism — expert parallelism (EP, with DeepEP kernels) and context parallelism (CP) for long-sequence training — plus selective activation checkpointing, FP8 training on Hopper+, LoRA, and multi-tenant training (many concurrent LoRA tenants sharing one trainer + inference deployment). You can read more in the dedicated [training documentation](training.md).

The three processes communicate through configurable transports — by default the trainer↔orchestrator rollout link uses the local filesystem, and weight broadcast uses the filesystem (or NCCL for synchronous setups). Swap to ZMQ for multi-host setups without shared storage. See [Scaling](scaling.md) for the deployment options.

## Installation

Install this repository and its pinned dependencies using the [Prime Values setup instructions](../README.md#setup). Installation requires no private infrastructure.

## Quick Run

The supported native reverse-text policy example uses the public Qwen3-4B model, groups of eight, three updates, and two GPU roles. Choose hardware with sufficient memory; GPU count alone does not establish capacity. Inspect its resolved configuration first:

```bash
uv run rl @ configs/native/reverse_text_policy.toml --dry-run --output-dir outputs/native-check
```

Then launch on a provisioned machine:

```bash
WANDB_MODE=offline uv run rl @ configs/native/reverse_text_policy.toml --output-dir outputs/native-smoke
```

The `rl` entrypoint resolves per-role configs and launches inference, orchestrator and trainer. The [native profile](../configs/native/README.md) also provides an independent-value example with four GPU roles and cached RG variants. These are bounded lifecycle examples, not convergence benchmarks.

## Documentation

- **[Configuration](configuration.md)** — TOML composition, CLI overrides, dry-run.
- **[Training](training.md)** — Launch and observe RL and SFT runs.
- **[Inference](inference.md)** — vLLM-backed server (or fleet) holding the current policy.
- **[Scaling](scaling.md)** — Single-GPU through multi-node clusters via FSDP / EP / CP and SLURM.
- **[Algorithms](algorithms.md)** — Async semantics, loss / advantage / filter plugins, trajectory merging.
- **[Advanced](advanced.md)** — Custom modeling, multimodal, LoRA, multi-tenant, P/D inference.
- **[Development](development.md)** — Test suite, pre-commit hooks, adding a new model.
