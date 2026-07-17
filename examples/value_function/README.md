# Async value-function example

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
to generate. This example uses pure per-token GAE from the value function for
policy advantages and, by default, trains once on each one-batch replay cohort.
Raising
`value_function.replay.max_updates_per_rollout` grows the default replay and
allows uniform reuse without repeating one fixed batch. If the baseline were
omitted, enabling the value function would select the same pure-value baseline.
The default value head is two-bin classification over `[0, 1]`; see
[`docs/value-functions.md`](../../docs/value-functions.md) for regression,
separate policy/target lambdas, monitoring, and scaling options.
