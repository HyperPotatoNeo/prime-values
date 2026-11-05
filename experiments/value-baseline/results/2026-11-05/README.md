# November 5 2026 baseline results

Prime Values base: `30bfbf301335e8ce24c62fa5fd24bcc37a633bd3`. No production code or dependency changes. Runtime image and exact submodule revisions are in `provenance.json`; Python/CUDA/package/GPU versions are in `original/runtime.json`. Source archive builds report Verifiers/Renderers package metadata as `0.0.0`; their actual source is pinned by the submodule SHAs, not that metadata version.

## Observed results

- Original selected regression suite: **451 passed**, no failures/skips, 28.20 seconds.
- Expanded selected suite: **476 passed**, no failures/skips, 21.70 seconds. These are correctness runs, not a speed benchmark (compilation/cache state differs).
- New coverage: 15 independent return-oracle cases, 2 TETHER JSON-resume continuations, 5 analytical policy-gradient cases, 3 real tiny-model critic cases.
- Mutation control: **22 passed**. All **7 deliberately incorrect variants** were caught by assertion failures; no import/error/skip was counted as a kill. Details in `mutations/summary.json`.
- Real ValueTrainerRuntime/FSDP: one and two ranks each performed two updates with uneven action counts and a two-rank dummy microbatch. Initial/model/Adam moments and update deltas matched **exactly** between rank counts for this fixture.
- Fresh-process checkpoint resume: checkpoint1 restored **exactly**, and the next update reproduced uninterrupted step2 **exactly**, at both rank counts. Scheduler/progress also matched. See `distributed-comparison.json` and each rank's resume record.
- Ruff formatting/lint, shell syntax and Git whitespace checks passed.

One separate H200 node was reserved for the test pod (8 devices allocated; numerical tests used one or two). This was an intra-node check. The existing 80-GPU experiment was neither modified nor stopped. No new training quality or throughput claim follows from these results.

## Known defect exposed without freezing it

The diagnostic in `masked-nonfinite.json` confirms that an excluded token carrying NaN or either infinity makes the existing default policy loss nonfinite and leaves nonfinite gradients on the excluded position. The eligible token gradient remains finite in this fixture. Finite loss/gradients with zero excluded-token gradient are the desired behavior. This is recorded as an unresolved safety defect for the next change, not an expected passing value to preserve.

## Test setup corrections

Initial new model tests used FP32 standalone parameters under autocast, which promoted attention inputs to FP32 and failed FlashAttention's supported-dtype check. They were corrected to the actual BF16 serving-copy setup; categorical reference expectation now uses explicit FP32 arithmetic. No tolerance was widened. An early in-process resume probe hit the process-global monitor guard; the final probe uses a genuine fresh process instead. Failed attempt logs are retained in the evidence bundle.

## Interpretation and remaining gates

These results strengthen preservation checks for mathematical credit, loss gradients, packed inference/training, TETHER state, critic distributed normalization and critic restart. They do not prove arbitrary upstream changes safe. In particular:

- Production policy-trainer denominator construction is not executed by the analytical loss tests, which supply component counts explicitly.
- The real-model fixture is tiny Llama/FlashAttention2; no Qwen-specific, multi-node or alternative-backend parity claim.
- No fresh end-to-end policy generation/value-evaluator fanout run, replay pressure/cancellation stress test, or dedicated/colocated evaluator equivalence run was performed here.
- Future episode/staleness/payload changes still need dedicated topology, lifetime and complete-group admission checks.

Raw JUnit reports, logs, model/optimizer snapshots and checkpoints were preserved under `prime-values-shared:/data/value-baseline-001` and exported as `value-baseline-evidence.tar.gz` alongside the chat deliverables. Compact results are checked in here. The source archive used exact submodule commits; future runs should initialize those submodules before syncing the frozen lockfile. `final/runtime.json` fingerprints the pytest snapshot; `provenance.json` fingerprints the final standalone probe and additions.
