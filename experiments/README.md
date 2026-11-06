# Validation and experiment records

This index tracks the staged PrimeRL synchronization work while keeping Prime Values' independent value plane and learning contracts explicit. Detailed configurations, tests, results and limitations live with each milestone; bulky artifacts stay outside Git with recorded locations and checksums.

| Milestone | Purpose | Records |
| --- | --- | --- |
| Frozen value behavior | Establish independent numerical oracles, mutation sensitivity and real critic update/resume comparisons before production changes | [Methods](value-baseline/README.md), [November 5 results](value-baseline/results/2026-11-05/README.md) |
| RL safety and reliability | Selective upstream masking, diagnostics, checkpoint and failure-handling changes while preserving value semantics | [Scope and provenance](rl-safety/MILESTONE.md), [Validation](rl-safety/README.md), [Adversarial review](rl-safety/review.md), [November 5 results](rl-safety/results/2026-11-05/README.md) |
| Owned critic trajectories | Preserve critic geometry, credit and publication independently of policy sample mutation | [Scope](episode-sync/README.md), [Completed ownership evidence](episode-sync/results/ownership/README.md) |
| Canonical Episode producer | Adopt the actual native and maintained legacy producers through an explicit single-trace boundary | [Completed producer, numerical and real-model evidence](episode-boundary/README.md) |

The [initial upstream audit and staged plan](upstream-plan.txt) preserves the original feature assessment and value-contract analysis as a historical record.

Canonical Episode integration is complete for the recorded upstream landmark. The next unit moves task ownership into the orchestrator while preserving the value boundary, followed by packing/shared-runtime and policy-provenance changes. New objectives, staleness eligibility and aggregation alternatives require separate opt-in evaluation. These records establish covered behavior; they do not imply reward improvements or universal compatibility.
