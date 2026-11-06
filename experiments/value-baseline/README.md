# Value behavior baseline

This baseline protects intended value semantics before changing shared RL infrastructure. It combines independent numerical oracles, existing lifecycle tests, real model comparisons, and a real critic checkpoint/restart probe. Passing it is evidence about the covered contracts, not a certificate for every model or deployment.

The source baseline is Prime Values `30bfbf301335e8ce24c62fa5fd24bcc37a633bd3`. Production code, dependency pins, loss defaults, `optimization_dtype`, and `reduce_dtype` are unchanged by this test work.

## What is checked

| Check | Independent reference or contrast | Errors it should expose |
| --- | --- | --- |
| Policy and critic returns | Explicit finite-horizon mixture of n-step returns | Shared lambdas, discounting context, wrong terminal reward |
| Policy loss and gradient | Tabulated probabilities and analytical softmax derivatives | Wrong trust region, missing KL, lost fractional weights, incorrect loss routing or reference differentiation |
| TETHER continuation | Uninterrupted versus JSON-restored partial fitting window | Lost fitting statistics, changed application timing/state |
| Critic packing and serving | Separate real tiny Llama forwards and direct state-before-action loss | Cross-sequence attention, shifted value alignment, wrong normalization, reassembly mistakes |
| Critic distributed updates | Actual ValueTrainerRuntime at one versus two FSDP ranks | Wrong global token count, rank averaging, dummy microbatch effects |
| Critic resume | Fresh process restored at step one versus uninterrupted step two | Lost optimizer/scheduler/model/progress state |
| Existing behavioral suite | Existing branch/context, replay, transport, warmup and config tests | Regression in established supported contracts |

The distributed fixture has unequal sequence lengths and action counts, and an empty padding microbatch at two ranks. It compares Adam moments and parameter deltas, not only final model parameters: Adam can conceal a multiplicative gradient error in its first update. It also checks that moments and updates are nonzero.

## Run in an isolated environment

Use the checked-in lockfile and submodule revisions, with a compatible CUDA runtime. Do not run these commands in a pod containing live training: the existing `tests/conftest.py` cleanup fixture terminates matching torchrun processes. Do not run mutation pytest concurrently with the distributed probe. All commands run from this repository root.

```bash
uv sync --all-extras --frozen
bash experiments/value-baseline/run_checks.sh /tmp/value-baseline/checks
uv run --no-sync python experiments/value-baseline/check_mutations.py /tmp/value-baseline/mutations

uv run --no-sync torchrun --standalone --nproc-per-node=1 tests/integration/value_baseline_probe.py --output /tmp/value-baseline/rank-1
uv run --no-sync torchrun --standalone --nproc-per-node=1 tests/integration/value_baseline_probe.py --output /tmp/value-baseline/rank-1 --resume
uv run --no-sync torchrun --standalone --nproc-per-node=2 tests/integration/value_baseline_probe.py --output /tmp/value-baseline/rank-2
uv run --no-sync torchrun --standalone --nproc-per-node=2 tests/integration/value_baseline_probe.py --output /tmp/value-baseline/rank-2 --resume
uv run --no-sync python tests/integration/value_baseline_probe.py --compare /tmp/value-baseline/rank-1 /tmp/value-baseline/rank-2
uv run --no-sync python experiments/value-baseline/diagnose_masked_nonfinite.py
```

Use fresh output directories for each attempt. Preserve failed attempts separately. Model weights are generated locally from a seed; the probe needs no model download. The one/two rank check is intra-node, not an InfiniBand or multi-node qualification. Initial model states must be identical; a comparison requires completed one- and two-rank runs with matching fresh-process resume records.

`run_checks.sh` records package versions, GPU identity, imported source paths, source/test/config hashes and JUnit/log output. It rejects skipped/error cases. The mutation runner first requires a clean control, verifies actual copied-source imports inside each pytest process, and counts only assertion failures as killed mutations. Import errors, crashes, missing reports and timeouts are not successful detections. Mutation code lives in temporary copies; it never modifies production source.

## Numerical interpretation

Structural mappings and versions must match exactly. Analytical FP64 policy gradients use 1e-10 absolute/relative tolerance. The real model parity fixture uses BF16 serving-copy weights and FlashAttention2; production inference/optimization precision defaults remain untouched. Packed model loss/prediction tolerances are approximately 0.2%; per-parameter gradient L2 tolerance is 2% plus a small absolute floor. The distributed runtime uses the unchanged FP32 optimization/reduction defaults with BF16 compute; its one/two rank update and moment tolerance is 2.5% relative L2 plus 1e-6 absolute. Checkpoint load is exact; resumed continuation tolerance is 1e-5 relative L2 plus 1e-7 absolute.

These tolerances were specified before running comparisons. A failure requires investigating source, setup and numerical cause, not widening tolerance until green. Report observed errors with the runtime and source fingerprints. Reward improvement is not a correctness oracle for this baseline.

## Known gaps and deliberate limits

- The policy loss oracle supplies global component counts. It does not execute the policy trainer's construction of those counts in `trainer/rl/train.py`. Preserve token-count normalization explicitly in migration review; an end-to-end fractional-weight policy update remains a future gate.
- Tiny Llama coverage does not establish Qwen-specific behavior or every attention backend. The standalone runtime probe currently uses scalar MSE; categorical model loss/gradients are covered by the model fixture.
- No new live evaluator-weight fanout, full RL rollout loop, evaluator-placement equivalence, multi-node run or pressure/cancellation stress test is claimed here. Existing unit coverage is useful but not a substitute for those migration gates.
- Replay is intentionally cold after restart. The critic checkpoint probe does not claim exact replay/RNG restoration.
- The masked-nonfinite diagnostic reports whether excluded invalid log probabilities contaminate loss/gradients. Nonfinite output is a defect to fix separately, never a desired golden value. This diagnostic does not count as a passing behavioral test.
- Preserve the originally sampled action topology before any future policy staleness filtering. Today's tests do not validate upstream's not-yet-integrated episode/payload lifecycle.

When adding a feature, extend the smallest relevant oracle or real component probe. Do not replace independent expectations with outputs copied from the new implementation, remove failing cases, or conflate baseline compatibility with all new algorithms being equivalent.
