# Async value-function smoke

`rl.toml` runs all four roles on one four-GPU host with Qwen3-0.6B:

```bash
uv run rl @ examples/value_function/rl.toml --dry-run
uv run rl @ examples/value_function/rl.toml
```

The critic warms for two adopted evaluator versions while inference continues
to generate. It then uses a leave-one-out/value linear mixture for policy
advantages and reuses each newest critic batch twice.

For the four-node Qwen3-4B RG-Mix setup, use
`configs/rg_mix/async_value.toml` and overlay cluster-specific Slurm settings.
