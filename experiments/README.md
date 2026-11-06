# Validation and experiment records

This index tracks the staged PrimeRL synchronization work while keeping Prime Values' independent value plane and learning contracts explicit. Detailed configurations, tests, results and limitations live with each milestone; bulky artifacts stay outside Git with recorded locations and checksums.

| Milestone | Purpose | Records |
| --- | --- | --- |
| Frozen value behavior | Establish independent numerical oracles, mutation sensitivity and real critic update/resume comparisons before production changes | [Methods](value-baseline/README.md), [November 5 results](value-baseline/results/2026-11-05/README.md) |
| RL safety and reliability | Selective upstream masking, diagnostics, checkpoint and failure-handling changes while preserving value semantics | [Scope and provenance](rl-safety/MILESTONE.md), [Validation](rl-safety/README.md), [Adversarial review](rl-safety/review.md), [November 5 results](rl-safety/results/2026-11-05/README.md) |
| Owned critic trajectories | Preserve critic geometry, credit and publication independently of policy sample mutation | [Scope](episode-sync/README.md), [Completed ownership evidence](episode-sync/results/ownership/README.md) |
| Parameter-safe activation offload | Keep live parameter views on device while real activations offload | [Completed expert and compiled critic evidence](activation-offload/README.md) |
| Streaming trainer metrics | Bound entropy/mismatch storage while preserving metric populations and learning | [Completed trainer, distributed and profiler evidence](trainer-metrics/README.md) |
| Explicit packing boundaries | Prepare attention boundaries from owned host lengths across policy and critic paths | [Completed numerical and compiled-runtime evidence](packed-boundary/README.md) |
| Native task ownership | Select native task data once per group and reconstruct it on stateless workers, retaining legacy group scoring | [Completed task, producer and real-model evidence](task-ownership/README.md) |
| Canonical Episode producer | Adopt the actual native and maintained legacy producers through an explicit single-trace boundary | [Completed producer, numerical and real-model evidence](episode-boundary/README.md) |

The [initial upstream audit and staged plan](upstream-plan.txt) preserves the original feature assessment and value-contract analysis as a historical record.

Canonical Episode integration is complete for the recorded upstream landmark. Native task ownership is also complete. Explicit packing boundaries are also complete. Streaming entropy/mismatch collection is complete. Activation offload parameter ownership is complete. CPU-offloaded optimizer metadata restoration is being validated; exact token provenance requires a separately validated inference-stack upgrade. New objectives, staleness eligibility and aggregation alternatives require separate opt-in evaluation. These records establish covered behavior; they do not imply reward improvements or universal compatibility.
