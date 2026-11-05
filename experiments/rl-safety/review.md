# Safety and reliability source review

Reviewed November 5, 2026, against the working `feat/rl-safety-reliability` branch. This is a source and evidence-boundary review, not a declaration that live validation has completed. The reviewer implemented the reliability ports and independently reviewed the streaming diagnostics, pipeline counter checks and milestone claims; separate agents challenged the reliability implementation.

## Findings incorporated

- Converted-cache validation now strips LoRA-only keys, excludes the critic-owned head, and requires a source-present language-model head for both policy and critic consumers. This closes the identified partial-cache race for the same text backbone. Complete caches have a read-only fast path; incomplete generated children are rebuilt under a lock, with shard/index/key validation before publication.
- Orchestrator checkpoint serialization writes a temporary file before replacement, including adaptive algorithm state. Background component exceptions interrupt a blocked main loop and preserve the existing cleanup path. Failed initialized distributed ranks skip potentially blocking teardown; healthy shutdown remains unchanged.
- Streaming diagnostics retain four detached FP64 scalars per metric and use globally weighted token counts. Sorted key unions and neutral empty contributions align collectives when ranks have different populations. Explicit nonfinite flags survive aggregation; NaN extrema are not used as a validity oracle. All newly added diagnostic families use this path.
- The policy counter update moved before intermediate checkpoint writes without changing its arithmetic. Multi-run adapter progress is a separate, packer-owned object and is unaffected. The live harness checks actual persisted intermediate progress, cumulative packed-microbatch counts, and equality between saved and restored counters; final-only checkpoint checks would miss the original defect.
- No value mathematics, value head, replay/target construction, TETHER logic, model optimization/reduction precision or learning defaults changed. Diagnostic FP64 aggregation is separate from model precision.

## Evidence available at review

The coordinating agent reported the final selected suite passed 495 tests without skips, seven deliberate mutations were assertion-killed after a 22-case clean control, and focused reliability/summary tests passed. Reported distributed checks covered fatal exit, healthy shutdown, one/two-rank policy loss and gradient oracles, uneven diagnostic populations and nonfinite flags. Reported critic comparisons matched initial and two updated snapshots across milestones at both rank counts, including model and optimizer state, and matched fresh-process resumes exactly. These execution reports must be accompanied by their archived logs, source hashes and result manifest; this reviewer did not rerun them.

The exact-source Qwen live pipeline was still running at review time. No claim of completed dedicated/colocated lifecycle validation, intermediate live checkpoint verification or end-to-end milestone acceptance is made here. No new tests were run for this documentation review. Earlier local checks of the counter move passed Ruff and `git diff --check`.

## Remaining limits

Cache checks establish structural completeness, not tensor-value integrity, and assume an immutable source plus matching text-backbone consumers. Arbitrary model subsets, VLM layouts and uncooperative older cache writers are not covered. A non-UTF8 corrupted index fails safely rather than rebuilding automatically.

Atomic progress-file replacement is not a transaction across workers and does not establish power-loss durability. Fatal-rank behavior covers synchronous exceptions with an initialized process group; it does not address every signal, async entrypoint or blocked logging sink. Task-supervision tests exercise actual asynchronous failures but substitute external resources, so they do not replace live lifecycle checks.

Finite mathematical compatibility is checked with analytical oracles and frozen behavior, not inferred from test counts. Extreme active uncapped ratios can still overflow, and loss masking cannot repair upstream invalid activations. Live smoke runs establish lifecycle behavior, not reward improvement, placement equivalence, multi-node behavior, all-model compatibility or replay/RNG restoration.

The next canonical episode migration remains separate. It needs an explicit value-owned snapshot, preserved action/group/version semantics and a defined policy for multi-trace episodes before broad transport changes or new objectives are enabled.

## Recovery harness follow-up

Later live evidence showed completed training followed by a vLLM shutdown `EngineDeadError`. Reviewed the narrow classifier against the saved inference and launcher logs: training completion precedes inference shutdown, the only permitted traceback class is `EngineDeadError`, and application shutdown must complete after every traceback and terminal exception. Other component tracebacks remain failures. Independent replay accepted the actual cleanup and rejected an appended second `EngineDeadError` after shutdown completion. Review also required a finite actor gradient norm for every logged policy update; the harness now enforces that coverage.

Continuing from the existing fresh-stage checkpoints is acceptable as qualified recovery evidence. It rechecks requested/resolved and actual child configurations, finite losses and gradients, intermediate checkpoint counters, evaluator versions, completion markers and checkpoint files. Resumed stages must record a natural zero exit and restore the saved counters. Historical fresh stages without an exit receipt remain explicitly qualified: completed training/checkpoints do not establish an unrecorded natural process exit. This follow-up approves the validation approach, not the outcome of the resumed runs, which remained pending at review.

## Final continuation audit

Reviewed the subsequent launch-generation fix: before starting a stage, the harness moves existing live logs into that stage's `prelaunch-logs` archive. The launcher recreates the live log directory before starting children, so the scanner sees only the current generation. Prior stage evidence stays intact and traceback classification is unchanged. This corrects a false failure caused by reading an old shutdown traceback before the new launcher had started; it does not waive a current training error.

The coordinating agent subsequently reported trainer-colocated placement completed fresh policy steps 1–3 and resumed step 4 with natural resumed exit zero, while the independent critic continued from version 9 to 12. Final result records and archived runtime evidence remain the authoritative execution record; no local result JSON was available to this reviewer for independent replay. Dedicated placement had completed its fresh training, but its resumed retry was still starting after the stale-log failure was archived. Dedicated completion is not claimed here.

No additional source-level blocker was found. Continuation still validates the original configurations, intermediate counters, losses, actor gradient norms, evaluator observations, checkpoint files and restored progress. A failed partial resume that removed the required fresh checkpoint would fail these preconditions. This remains qualified lifecycle evidence: missing historical exit receipts are not reconstructed, archived failed attempts remain failures, and neither independent placement equivalence nor numerical learning quality follows from the smoke runs.

## Final result audit — November 5, 2026

Read the actual [dated result record](results/2026-11-05/README.md) and its pipeline, test, mutation, critic, distributed, shutdown-control and manifest JSON files. Both placements completed policy steps 1–3 followed by a fresh-process step 4; both resumed stages recorded natural exit zero. Dedicated critic versions progressed 4→5 and trainer-hosted versions 9→12. Both preserved their exact step-three counters across restart. Policy losses, actor gradient norms and critic losses in both summaries are finite. The fresh-stage exits remain explicitly unknown, matching the result record's qualification. These completed results supersede the earlier pending-runtime status in this review.

The record consistently reports 514 distinct passing cases without skips, seven assertion-killed mutations, eleven shutdown controls, exact tiny-critic comparisons and the bounded fatal-rank results. Independently recomputed all 313 recorded executable source/test/config/harness hashes against the reviewed checkout: every hash matched. The manifest records verification of the archive checksum and 445 file hashes, deletion of the test jobs and zero remaining test pods; those archive/cluster checks were performed by the coordinating agent, not repeated here. No tests were rerun during this final evidence audit. No inconsistency or additional blocker was found in the bounded result claims; the numerical, lifecycle and artifact limitations above still apply.
