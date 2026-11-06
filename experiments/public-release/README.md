# Public native-profile validation — November 6, 2026

The public Prime Values sync supports native reverse-text and cached RG with policy-only or independent-value training. It preserves the separate critic trainer/evaluator, value-owned trajectories, independent return parameters, and existing precision settings. Organization-specific deployment tuning and the future internal experiment version are separate work.

## Executed checks

The runtime source and lifecycle test were pinned to `6e3a8625fae969c22b82e7d7a309b52bce0a0d28`. The later acceptance-documentation commit does not change that implementation.

| Check | Actual result |
| --- | --- |
| Native policy, Qwen3-0.6B, group eight | Three fresh updates, resume to four, both launchers exit normally; saved progress restored and nonzero finite gradients |
| Native independent value, Qwen3-0.6B, group eight | Three fresh actor updates, resume to four; independent critic checkpoint advances from four to five; both launchers exit normally |
| Native RG/value, Qwen3-4B-Instruct-2507, group eight | Three fresh actor updates and resume to four, both launchers exit normally with finite losses and gradients |
| Selected GPU policy-loss and fused-head tests | 35 passed, five deselected |
| Hosted Linux CPU suite at the runtime revision | 925 passed, one skipped, 105 GPU tests deselected; isolated slim installation and style also pass |
| Public RG artifact generation | Real six-row multiprocessing generation/save; native train/eval split loading and all six correct-answer scores verified |

These GPU checks ran sequentially on H200 hardware. Native reverse-text uses a 512-token sequence budget and 64 output tokens; RG uses 2,048/128. All retain groups of eight, ancestral sampling, and normal reward/filter logic. The lifecycle checks use NCCL weight publication and actual model/optimizer checkpoints. The RG actor peaked at approximately 78 GiB: a GPU role count alone does not establish memory fit.

The resolved model revisions were `c1899de289a04d12100db370d81485cdf75e47ca` (0.6B) and `cdbee75f17c01a7cc42f958dc650907174af0554` (4B). The existing RG artifact has 10,000 training rows and 256 held-out rows; its file hashes, scoring dependency and effective configuration are recorded with the evidence. This smoke did not run held-out evaluation.

## Failure retained and corrected

The original small-model policy test exhausted its 64-token budget on reasoning and produced ten consecutive empty post-filter batches. The unchanged orchestrator correctly stopped. Eight actual value-run outputs from the same model/settings corroborated the diagnosis: reasoning-only, length-limited responses with no answer content. The discarded policy cohorts were not serialized and are not claimed as recovered.

The small-model test now explicitly selects Qwen3's non-thinking prompt mode. All 27 original assertions remain, with an added check for that prompt mode. No reward, filtering, sampling, seed, token budget, public model default or learning objective was changed to obtain a pass. All four real config/request/child-CLI preflights and the corrected GPU tests passed.

Independent review rescored all 64 recorded corrected native rollouts exactly, confirmed reward variation and nonzero gradients, and verified warmup and resumed critic adoption. At the tested unit-discount/unit-lambda settings, token-weighted rewards agree with the recorded value targets and prediction-plus-advantage means within `1e-12`. The earlier [frozen numerical](../value-baseline/README.md), [compiled critic/backend](../backend-sync/README.md), and [actor observation-only](../provenance-diagnostics/README.md) controls remain the numerical-preservation evidence.

## Limits and reproducibility

The short RG smoke truncates responses and is not a learning-quality or convergence result. Its nearly constant targets also make explained variance ill-conditioned; that diagnostic is not evidence of learning quality. Choose a suitable generation budget for substantive reasoning experiments. The RG policy-only template was not GPU-validated: the observed short-budget RG groups had uniform rewards, which normal mean-baseline filtering would remove. That template needs task/budget selection and its own runtime check before use. Replay refills from new rollouts after resume; task-source/replay cursors are not restored, so an uninterrupted trajectory is not reproduced exactly.

H200 execution does not establish A6000 memory fit, arbitrary hardware/model/topology compatibility, the full historical environment catalog, or universal old-checkpoint compatibility. FA4 remains deferred. Self-hosted GPU Actions are opt-in and skipped without configured runners; those skipped jobs are not acceptance evidence.

Run the public native lifecycle checks with `uv run pytest tests/integration/test_native_profile.py -vvs`. The [native configurations](../../configs/native/README.md) and [RG generator](../../packages/rg-mix-env/README.md) require no private infrastructure. Full raw logs, failed attempts, configuration/source hashes, independent reviews and resource receipts are retained in the supplementary private [research record](https://github.com/resolutionorg/siddarth-research-cache/tree/main/projects/prime-values/records/2026-11-06-public-release). Full model/optimizer checkpoint trees were used during validation but are not archived; checkpoint progress/configuration receipts and source/model/data hashes are retained.
