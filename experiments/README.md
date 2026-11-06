# Validation and experiment records

This index tracks the staged PrimeRL synchronization work while keeping Prime Values' independent value plane and learning contracts explicit. Detailed configurations, tests, results and limitations live with each milestone; bulky artifacts stay outside Git with recorded locations and checksums.

| Milestone | Purpose | Records |
| --- | --- | --- |
| Frozen value behavior | Establish independent numerical oracles, mutation sensitivity and real critic update/resume comparisons before production changes | [Methods](value-baseline/README.md), [November 5 results](value-baseline/results/2026-11-05/README.md) |
| RL safety and reliability | Selective upstream masking, diagnostics, checkpoint and failure-handling changes while preserving value semantics | [Scope and provenance](rl-safety/MILESTONE.md), [Validation](rl-safety/README.md), [Adversarial review](rl-safety/review.md), [November 5 results](rl-safety/results/2026-11-05/README.md) |
| Owned critic trajectories | Preserve critic geometry, credit and publication independently of policy sample mutation | [Scope](episode-sync/README.md), [Completed ownership evidence](episode-sync/results/ownership/README.md) |
| Exact-order packing refinement | Vectorize CPU swaps with identical assignments, batches and counts | [Completed CPU parity and timing evidence](packing-swap/README.md) |
| CPU-offloaded optimizer resume | Restore saved optimizer-group metadata without replacing parameter/state buffers | [Completed actual DCP and continuation evidence](optimizer-resume/README.md) |
| Parameter-safe activation offload | Keep live parameter views on device while real activations offload | [Completed expert and compiled critic evidence](activation-offload/README.md) |
| Streaming trainer metrics | Bound entropy/mismatch storage while preserving metric populations and learning | [Completed trainer, distributed and profiler evidence](trainer-metrics/README.md) |
| Explicit packing boundaries | Prepare attention boundaries from owned host lengths across policy and critic paths | [Completed numerical and compiled-runtime evidence](packed-boundary/README.md) |
| Native task ownership | Select native task data once per group and reconstruct it on stateless workers, retaining legacy group scoring | [Completed task, producer and real-model evidence](task-ownership/README.md) |
| Canonical Episode producer | Adopt the actual native and maintained legacy producers through an explicit single-trace boundary | [Completed producer, numerical and real-model evidence](episode-boundary/README.md) |
| Native RG and producer runtime | Preserve observed RG tasks/scoring and move an explicit native profile to the newer Episode/renderer APIs | [Native RG](native-rg/README.md), [Sequence boundary](sequence-boundary/README.md), [Producer acceptance](native-producer/README.md) |
| Sampled token versions | Preserve actual physical sampled-token provenance through clipping, packing and transport | [Carrier acceptance](weight-versions/README.md) |
| Inference/training backend | Adopt vLLM 0.32/Torch 2.13 with exact compiled critic controls and real weight adoption/restart | [Backend acceptance](backend-sync/README.md) |
| Sampled policy coverage and age | Report known, unknown and future labels without changing losses or token eligibility | [Diagnostics acceptance](provenance-diagnostics/README.md) |
| Mismatch by sampled weight age | Identify mismatch concentration in older sampled tokens without changing token selection | [Detached aggregation and distributed checks](mismatch-by-age/README.md) |
| Public profile closure | Verify public setup, artifact generation, real policy/value/RG fresh-resume and focused GPU losses | [Executed checks and limits](public-release/README.md) |
| Native launch and compatibility contracts | Provide usable RG/reverse policy/value configs and explicit historical compatibility boundaries | [Native launch guide](../configs/native/README.md), [Complete config inventory](dependency-migration/config-compatibility.json) |

The [initial upstream audit and staged plan](upstream-plan.txt) preserves the original feature assessment and value-contract analysis as a historical record.

The recorded native single-trace profile has completed producer, backend and sampled-weight attribution acceptance, alongside the earlier value-ownership and execution improvements. Numerical fixtures, real training/restart evidence and known limitations are recorded per unit. These results do not imply reward improvements or universal compatibility.

[PR #8](https://github.com/HyperPotatoNeo/prime-values/pull/8) collects these units for the public native profile. The owner selected native RG mix and reverse-text with the historical runtime retained separately. The [final review record](https://github.com/resolutionorg/siddarth-research-cache/tree/main/projects/prime-values/records/2026-11-06-pr-review) records adversarial findings, native test-fixture repairs, published-diff checks and observed CI outcomes. Native GPU lifecycle checks are distinct from historical convergence tests; unexecuted checks are not acceptance evidence.

The [environment inventory and dependency assessment](dependency-migration/README.md) remain the reference for compatibility outside that profile. Existing legacy configurations must not be silently mapped to different native tasks. New objectives, staleness eligibility, adaptive scheduling and sampling-mask replay remain separately scoped work rather than implicit changes to training defaults.
