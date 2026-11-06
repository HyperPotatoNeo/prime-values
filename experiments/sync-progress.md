# Upstream synchronization status

The value-function implementation already existed in Prime Values. This work ports selected Prime RL infrastructure changes while preserving its independent critic trainer, evaluator, replay, credit construction and learning defaults. It is not a merge of every upstream commit.

The comparison target is Prime RL `3c96816de9a2a906649a04026a46517840a0f4a6`; the fork started from `30bfbf301335e8ce24c62fa5fd24bcc37a633bd3`. The original comparison remains in [the archived plan](upstream-plan.txt). Current evidence is indexed in [the experiment records](README.md).

## Completed units

| Unit | Result | Source |
| --- | --- | --- |
| Behavioral baseline | Independent mathematical/gradient checks, mutation controls, real critic update and restart evidence | `95263173` |
| Safety and reliability | Excluded-token arithmetic safety, bounded tail diagnostics, checkpoint counters, atomic publication and visible component failures | `f6b9a40c` |
| Critic ownership | Critic-owned trajectory snapshots and pure credit module, independent of later policy mutation | `4b8ca528` |
| Canonical Episodes | Actual native and maintained legacy producers through a deliberately single-trace boundary | `4835142d` |
| Explicit packing boundaries | Host-owned boundaries with exact covered model, CP and compiled critic comparisons | `1708eb0c` |
| Streaming trainer metrics | Bounded standard entropy/mismatch collection; preserved scalar/custom metrics | [Evidence](trainer-metrics/README.md) |
| Native task ownership | One selected/serialized task per group; stateless native workers; preserved legacy addressing and group scoring | `5f80407b` |

Each unit has its own evidence and limitations. In particular, the earlier safety unit's historical fresh-exit uncertainty remains in its record; later normal-exit observations do not rewrite it. Passing these checks does not establish improved reward, all model families or exact stochastic replay after restart.

## Current efficiency work

Explicit packed-sequence boundaries have passed their separate acceptance checks and are included in this unit. Existing host sequence lengths supply attention boundaries for supported custom FlashAttention models in policy training, critic training and the shared value-serving helper. Padding, token alignment, independent GAE/TD lambdas, targets, loss normalization and optimizer/reduction dtypes remain unchanged. Acceptance compared actual outputs, full gradients and updates to frozen source, including context parallelism and the normal compiled distributed critic runtime; all covered paired tensors matched exactly. See [packing evidence](packed-boundary/README.md).

Standard entropy and mismatch now use masked GPU summaries instead of retaining per-token CPU arrays and gathering all tokens. Scalar/sample/custom metrics retain their existing path and medians. Stable variance, nonfinite values, empty ranks and environment membership passed independent CPU and NCCL oracles. An actual trainer update preserved exported weights and filtered telemetry; the accumulation profiler confirmed reduced host transfers without establishing end-to-end speed. See [metric evidence](trainer-metrics/README.md).

The next bounded unit fixes activation-offload ownership: saved views of live parameter storage must remain on device while real activations still offload. Compiled/FSDP storage lifecycle and full critic gradients/updates require acceptance before integration.

## Next dependency boundary

Exact sampled-token policy ages require a coordinated producer chain: vLLM's version-labeled generation, renderer pass-through and the canonical verifier schema. The current stack does not produce that information. Group-start policy versions remain useful scheduler provenance but must not be labeled exact token ages. A consumer that reports only unknown values is not considered a completed feature.

The [provenance review](policy-provenance-review.json) identifies the exact upstream commits, compatibility seams and acceptance gates. A future stack upgrade must preserve environment compatibility, token/prompt alignment and value behavior before adding observation-only token ages and mismatch bins. Stale-token masking, changed admission and new policy objectives remain separate changes. Neither this plan nor the existing cache authorizes changing optimization/reduction precision or silently adopting upstream loss defaults.

Transport reorganization, new model families, specialized kernels and new objectives are not prerequisites for the completed units. Import them only when they provide a concrete benefit with a clear value-preservation boundary. Keep source changes on the synchronization branch; do not merge Prime Values main as part of validation.
