# Native task ownership validation

This unit follows Prime RL `f5946b098e7d085732897292c150e959138bb703` and verifier `b13ba60da63cea91389e7575766b7270d0d11fc5`. The orchestrator selects native task data once and owns one serialized payload per group. Stateless workers reconstruct that payload; the legacy bridge keeps its dataset row addressing and joint group scoring. The Episode adapter and value-owned snapshot remain unchanged.

Finite selection retains the previous seeded dataset-position sequence even when semantic task IDs are gappy or repeated. Selected tasks require integer IDs for existing records, the declared task class, the configured task settings, and lossless data reconstruction. Unsupported per-row subclasses/configuration fail before execution. Task behavior must come from this declared class, data and config; incidental instance attributes are not a wire contract. Infinite tasksets advance per admitted group, and evaluation requires an explicit bound. Source RNG, cursor and generator state are not checkpointed; exact task-order continuation after restart is not claimed.

`test_task_ownership.py` checks the recorded old sampler sequence, payload ownership under nested mutation, completed/in-flight/unscheduled cancellation, bounded infinite pulls, and rejected reconstruction losses. Its transport boundaries are small deterministic substitutes. `producer_smoke.py` separately runs actual native/legacy ZMQ producers and a spawned native harness with a local deterministic HTTP endpoint. Neither proves model quality or GPU numerical equivalence.

Run the standalone preflight from the repository, with the native fixture importable:

```sh
PYTHONPATH=src:experiments/task-ownership uv run python experiments/task-ownership/preflight.py --output /fresh/evidence/path
```

It creates actual native servers in two spawned processes, verifies one dataset load on the client and none in workers, compares complete nested/private payloads after source mutation, and compares local grading artifact hashes. Harbor checks use its actual pinned parser and runtime configuration resolver, including restricted network metadata and explicit failure on an unsupported runtime. This does not test Docker's network enforcement or an external filesystem.

Harbor workers need their selected `task_dir/tests` artifacts on the same host. Local spawned workers are supported; externally addressed Harbor workers fail before connection because shared grading-artifact paths have not been verified. The verifier advancement also intentionally adopts Harbor dataset network metadata: restricted tasks require Docker policy support. No override silently restores unrestricted networking. Existing local paths and unrestricted tasks retain their contract, but arbitrary Harbor environments are not claimed equivalent.

Dependency scope: only the verifier gitlink advances for this unit. The pinned renderer, research environments, numerical packages, policy/value dtypes, loss and value prompt construction remain fixed. Current source comparisons and bounded GPU checks must be recorded alongside the final integration receipt; staging preflight success alone does not authorize a numerical-equivalence claim.

## Completed integration evidence

The active branch passed 47 focused task/producer checks and 3 configuration checks, plus 8 exact comparisons against frozen `f6b9a40c` for credit, loss, full-logit gradients and TETHER continuation. CPU receipts and the unchanged baseline hash are in `results/receipt.json`. Independent review covered group cancellation, serialized task ownership, reconstruction losses and Harbor admission.

A dedicated Qwen3-4B-Instruct-2507 run (model revision `cdbee75f17c01a7cc42f958dc650907174af0554`) completed policy steps 1–3 and critic updates 1–4, then restarted and completed policy step 4 and critic update 5. Both stages exited naturally with code 0; all classified component logs were clear. Resume restored 2,864 tokens / 6 samples exactly and finished at 3,860 tokens / 8 samples. This is a bounded integration check, not a reward-improvement or exact stochastic-restart claim. It supplements the preceding Episode unit's checks of both critic placements and value-off execution.

`results/gpu-summary.json` records losses, gradient norms, adoption and checkpoint progress. The parent independently verified all 132 archived member hashes and 320 runtime code/config/test hashes against the active source; the node also verified 2,342 source entries after the driver completed. Policy requests were separately audited as ancestral sampling (temperature 1, top_p 1, top_k disabled, min_p 0, penalties 0/0/1); this is source/config/log evidence, not a wire capture. Historical receipts were not rewritten.

Raw evidence: `prime-values-shared:/data/task-ownership-001/evidence.tar.gz`, 284,864 bytes, SHA256 `6f2870cafcb541f27c610f35fc418c9048e5d1abd964b4bdb99d800438b9d881`. The archive and sampling supplement are also preserved in the Resolution research cache's task-ownership record. Full model/optimizer checkpoint tensors are excluded; this archive alone cannot resume training. GPU allocation accounting is separate from measured utilization; see `../gpu-accounting/2026-11-05-sync-usage.json`.
