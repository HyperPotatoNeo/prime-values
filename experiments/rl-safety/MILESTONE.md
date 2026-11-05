# Prime Values safety and reliability milestone

This is the first production integration after the frozen value-behavior baseline. It selectively adapts useful PrimeRL safety and reliability changes to Prime Values. It does not merge the newer episode/transport architecture or change learning defaults.

## Scope and provenance

Prime Values base: `30bfbf301335e8ce24c62fa5fd24bcc37a633bd3`, followed by baseline commit `95263173`. Upstream was inspected at `3c96816de9a2a906649a04026a46517840a0f4a6`. These are adapted changes, not whole-commit cherry-picks.

| Change | Upstream reference | Prime Values contract |
| --- | --- | --- |
| Select eligible and accepted tokens before unsafe arithmetic; stable probability difference | `e3dd6763` | Keep asymmetric DPPO, optional IPO, reference-KL, coefficients and uncapped active policy ratios |
| Importance-ratio and log-ratio tails, accepted/all populations, saturation and nonfinite indicators | `f7e88f73`, reviewed with `e3dd6763` | Diagnostic cap only; fractions use token counts, independent of loss weights |
| Atomic orchestrator checkpoint publication | `b127f037` | Preserve progress and adaptive algorithm state in the existing checkpoint format |
| Propagate failed background components promptly | `8dede158` | Preserve existing orchestrator cleanup and independent value lifecycle |
| Failed distributed rank exits without waiting in collective cleanup | `855a2b81` | Healthy shutdown still destroys the process group normally |
| Validate and rebuild incomplete converted-model caches | `0c199a75` | Serialize independent policy/critic masters, publish staged complete caches, preserve read-only complete caches |
| Include completed-step counters in intermediate policy checkpoints | `80385dd9` | Move existing RL accounting only; no SFT or precision changes |

Additional integration work extracts the existing token-count denominator into one shared helper so the real trainer and distributed analytical probe exercise the same reduction. It does not switch normalization to the sum of fractional weights.

## Value boundary

The value mathematics, value head, independent trainer, evaluator, replay, temporal masks, target construction, lambda choices and TETHER update/state logic remain owned by their existing modules. No value loss or optimizer state moves into the policy trainer. Shared model loading validates base weights without treating a critic head or LoRA adapters as required pretrained cache keys.

Model `optimization_dtype` and `reduce_dtype`, their defaults, and critic update precision are unchanged. The FP64 change below applies to diagnostic aggregation, not optimization.

Excluded NaN/Inf loss inputs are an explicitly corrected baseline defect, not behavior to preserve. For finite eligible inputs, the established scalar and full-logit gradient oracles remain the compatibility checks. Accepted, nonzero policy terms remain uncapped and can still overflow for genuinely extreme active inputs; this patch does not silently change that objective. Loss-level masking cannot repair invalid activations upstream of the loss.

## Diagnostic cost and meaning

New diagnostics retain four FP64 scalars per metric and reduce count/sum/min/max across ranks. Their means are globally token-weighted, including uneven and empty rank populations. They do not gather full-step diagnostic arrays or retain Python lists of every token. Temporary arrays remain bounded by a microbatch. New diagnostics intentionally do not report medians or standard deviations. Existing distribution metrics continue through their existing collector, with a consistent FP64 collective dtype to prevent rank dtype mismatches and large-diagnostic variance overflow.

The ratio and mismatch reporting limit is `1e30`. Raw log ratios and explicit saturation/nonfinite fractions expose the tail. Undefined empty-population statistics and nonfinite active inputs remain visible. Distributed extrema involving NaNs are backend-dependent; use the explicit nonfinite fraction rather than interpreting a finite extremum as proof of validity.

## Review findings incorporated

Independent implementation/review passes challenged numerical semantics, cache concurrency and actual pipeline validation. They caught a diagnostic tensor retaining a gradient graph, extreme capped-KL subtraction producing a negative result, shared-cache LoRA/head-key mismatches, and excessive diagnostic gathering at production batch sizes. These were corrected before the current live validation attempt. Seven deliberate mathematical/data mutations remain detectable by assertions rather than import errors or crashes.

The live harness also needed corrections: the evaluator version endpoint uses MessagePack; native tasksets require the native harness; optional `None` config values disappear in generated TOML. The harness now instantiates the real environment server before GPU launch, reloads generated child configs through actual CLI parsers, recursively checks component logs, and retains failed attempts separately. It uses existing compilation/activation-checkpoint defaults rather than claiming a disabled setting that the launcher would restore.

## Evidence and limits

See [validation commands](README.md) and the dated result record for executed checks, exact configuration, source hashes and artifact locations. Frozen critic snapshots are compared directly across milestones, alongside one/two-rank and fresh-process resume comparisons. A short Qwen pipeline checks lifecycle, warmup, publication and resume; it is not a reward or throughput benchmark.

The [November 5 result record](results/2026-11-05/README.md) reports completed numerical, mutation, distributed, frozen-critic and both-placement restart checks. Historical fresh-stage exit codes remain explicitly unknown; both resumed stages recorded natural exit zero. See [review scope and limits](review.md) for the independent audits and qualifications.

Cache validation checks safetensor structure, keys and index mappings, not tensor-content hashes. It assumes an immutable source and the same text backbone for policy and critic. Atomic checkpoint replacement protects file visibility, not power-loss durability or a transaction across all workers. The fatal-rank fix covers synchronous exceptions after process-group initialization. Multi-node execution, evaluator-placement numerical equivalence, replay/RNG restoration and every model family remain outside this milestone.

## Next substantial unit

Migrate the canonical episode and shared-runtime boundaries with all new learning features off. Define an explicit value-owned snapshot before policy eligibility filtering or payload release; preserve original action topology, complete leave-one-out groups, evaluator-version warmup barriers and independent critic clocks. Specify the value unit for episodes containing multiple traces before enabling them; reject ambiguous cases until their credit assignment is defined. Then add bounded sampling replay and policy provenance, and evaluate staleness masking or new objectives separately.
