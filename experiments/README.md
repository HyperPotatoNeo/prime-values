# Validation and experiment records

This index tracks the staged PrimeRL synchronization work while keeping Prime Values' independent value plane and learning contracts explicit. Detailed configurations, tests, results and limitations live with each milestone; bulky artifacts stay outside Git with recorded locations and checksums.

| Milestone | Purpose | Records |
| --- | --- | --- |
| Frozen value behavior | Establish independent numerical oracles, mutation sensitivity and real critic update/resume comparisons before production changes | [Methods](value-baseline/README.md), [November 5 results](value-baseline/results/2026-11-05/README.md) |
| RL safety and reliability | Selective upstream masking, diagnostics, checkpoint and failure-handling changes while preserving value semantics | [Scope and provenance](rl-safety/MILESTONE.md), [Validation](rl-safety/README.md), [Adversarial review](rl-safety/review.md), [November 5 results](rl-safety/results/2026-11-05/README.md) |

The [initial upstream audit and staged plan](upstream-plan.txt) preserves the original feature assessment and value-contract analysis as a historical record.

The next planned milestone is the canonical episode/shared-runtime boundary, followed by sampling replay and policy provenance. New objectives, staleness eligibility and aggregation alternatives require separate opt-in evaluation. These records establish covered behavior; they do not imply reward improvements or universal compatibility.
