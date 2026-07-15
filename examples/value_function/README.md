# Async value-function smoke

`rl.toml` runs the default four-role topology on one four-GPU host with
Qwen3-0.6B:

```bash
uv run rl @ examples/value_function/rl.toml --dry-run
uv run rl @ examples/value_function/rl.toml
```

To serve values from the value trainer and reserve three GPUs instead, add one
override:

```bash
uv run rl @ examples/value_function/rl.toml \
  --value-function.evaluator.placement trainer
```

The critic warms for two adopted evaluator versions while inference continues
to generate. It then uses a leave-one-out/value linear mixture for policy
advantages and takes one optimizer update from each newest critic batch. The
default value head is two-bin classification over `[0, 1]`; see
[`docs/value-functions.md`](../../docs/value-functions.md) for regression,
separate policy/target lambdas, monitoring, and scaling options.

For the four-node Qwen3-4B RG-Mix setup, use
`configs/rg_mix/async_value.toml` and overlay cluster-specific Slurm settings.
