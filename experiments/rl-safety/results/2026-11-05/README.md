# RL safety and reliability — November 5, 2026

The selective safety/reliability milestone passed the covered numerical and lifecycle checks. Value mathematics, target construction, replay ownership, independent critic clocks, TETHER state, and model optimization/reduction precision are unchanged. This is a bounded compatibility result, not a full upstream merge or evidence of improved reward.

See [scope and upstream provenance](../../MILESTONE.md), [reproduction commands](../../README.md), and [adversarial review](../../review.md). The branch builds on frozen baseline `95263173` and Prime Values main `30bfbf30`; upstream was inspected at `3c96816d`.

## Numerical and failure evidence

| Check | Observed result |
| --- | --- |
| Selected baseline, policy and reliability tests | **514 distinct cases passed**, zero failures/errors/skips; 495 baseline cases plus 19 distinct focused cases |
| Deliberately incorrect implementations | All **7 mutations** detected by assertion failures after a 22-case clean control |
| Real tiny-critic updates at one and two ranks | Model parameters, optimizer moments and updates match exactly across ranks and fresh-process resume |
| Direct comparison to frozen baseline | Exact initial, step-one and step-two model/optimizer snapshots at both rank counts; 22 model keys and 66 optimizer entries after updates |
| Policy denominator and full gradient | One/two-rank analytical checks passed with fractional weights, separate components and empty populations |
| Diagnostics | Unequal rank populations, absent/empty keys, large ratios and explicit nonfinite fractions passed; new summaries store four scalars per metric |
| Excluded nonfinite inputs | NaN and positive/negative infinity at excluded positions produce finite loss and gradients |
| Failed distributed rank | Baseline timed out after 45 seconds; fixed implementation exited with failure in 10.29 seconds in the final run; healthy shutdown exited zero |
| Shutdown classifier | Eleven controls on actual saved logs passed, including rejection of genuine/early/additional/incomplete exceptions |

[Test counts](test-summary.json), [mutation outcomes](mutation-summary.json), [critic comparisons](critic-comparison.json), [frozen one-rank](frozen-comparison-1.json), [frozen two-rank](frozen-comparison-2.json), [distributed outcomes](distributed-final.json), and [shutdown controls](shutdown-controls.json) retain the machine-readable results. The 18-case focused report and three-case final summary report overlap in two cases; they are not counted twice.

The mutations challenge shared GAE/TD lambdas, context treated as action, symmetric DPPO, removing KL from rejected actions, differentiating the reference advantage, gating CE by the action mask, and unshifted value states. Tests use independent mathematical expectations and actual model/checkpoint behavior. These checks supplement source review; their passing is not treated as exhaustive proof.

## Live Qwen pipeline

Both placements used actual Qwen generation, policy training, independent critic training and evaluator serving. Fresh runs completed three policy updates; new processes resumed policy step three and the latest independently saved critic version, then completed policy step four.

| Placement | Policy restart | Critic restart | Saved/restored policy counters | Final policy loss / gradient norm | Resumed launcher exit |
| --- | --- | --- | --- | --- | --- |
| Dedicated evaluator | 3 → 4 | 4 → 5 | Step 3: 2844 tokens, 6 samples; step 4: 3840, 8 | −0.0038 / 2.0428 | 0 |
| Trainer-hosted evaluator | 3 → 4 | 9 → 12 | Step 3: 2870 tokens, 6 samples; step 4: 3842, 8 | −0.0082 / 2.0658 | 0 |

All logged policy and critic losses and policy gradient norms were finite. Warmup, evaluator-version progress, checkpoint publication, and policy/orchestrator/critic resume messages were verified. Intermediate step-one actor checkpoints contained completed-update counters: checking final checkpoints alone would have missed the original counter-lag defect. Counters count packed microbatches and padded tokens, not raw rollouts. Saved and restored step-three counters match exactly.

**Qualification:** the original fresh-stage launcher exit codes were not recorded. Their completed updates, logs, evaluator observations and checkpoints were reverified, but that does not reconstruct an unknown exit code. Both resumed stages have recorded natural exit zero. See [dedicated results](pipeline-dedicated.json) and [trainer-hosted results](pipeline-trainer.json).

Configuration: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`; native `reverse-text-v1`, null harness, 1000 tasks, sequence limit 512, completion limit 64, temperature/top-p 1, batch 8, group 2, two critic warmup updates, policy learning rate 1e-6, checkpoint interval/retention 1. Compilation, activation checkpointing and activation offloading retain existing defaults; optimizer offload is disabled. Model optimization and reduction precision remain FP32. Requested/resolved configurations and actual child CLI roundtrips are archived.

Each final test pod reserved eight H200s and 512 GiB host RAM; the local launchers used four GPU roles for dedicated serving and three for trainer-hosted serving. These are separate single-node runs, not a multi-node or placement-equivalence experiment. The unrelated training experiment was untouched.

## Failures and review corrections

Failed attempts are retained in the evidence archive. The first full-pipeline pod exceeded its 128 GiB host-memory limit during concurrent full checkpoints and was OOM-killed; replacement pods used 512 GiB without changing model precision. Other harness corrections addressed MessagePack version responses, the native environment contract, actual TOML/CLI config roundtrips, and the orchestrator checkpoint directory.

A completed fresh run emitted vLLM `EngineDeadError` during expected inference shutdown. The classifier now permits only that specific post-completion, post-shutdown-trigger case and requires final application shutdown after every terminal exception; other traces still fail. A dedicated restart attempt then encountered stale previous-stage logs before its launcher wrote anything. That failed attempt is archived; its fresh checkpoint manifest was rechecked unchanged before retry. Each new stage now archives previous live logs before launch. Neither correction widens numerical tolerances or changes learning behavior.

Independent review also caught excessive per-token diagnostic retention, a retained gradient graph, unstable extreme-tail diagnostic subtraction, and converted-cache key mismatches between policy/critic/LoRA consumers. Fixes were reviewed and covered by the final checks. No claim of universal cache/model-family compatibility follows.

## Reproducibility and artifact access

[Manifest](manifest.json) records exact image/model/submodule revisions and archive location/checksum. [Runtime source](runtime-source.json) records dependency versions and **313 source/test/config/harness hashes**, all compared to the reviewed local files. Verifiers and Renderers report archive metadata version `0.0.0`; use the submodule SHAs in the manifest. The final source hashes cover executable code and tests; prose/result records were completed afterward.

The exported archive is `rl-safety-evidence.tar.gz`, 26,891,157 bytes, SHA-256 `b75ae90602c9b0c792b09183b497c953b7585d4826a7b69a07d92004884055aa`. Persistent cluster storage: `prime-values-shared:/data/rl-safety-001/evidence.tar.gz`. The local export was verified against this checksum and all **445 individual file hashes** in its inventory. Cluster-storage access is separate from repository access; this archive is not hosted by GitHub.

The archive includes tests, mutations, failed/successful pipeline logs and configs, checkpoint metadata/counters, tiny-critic states and comparisons. It excludes full Qwen checkpoints/weights/broadcasts, source/cache directories and symlink targets. Full Qwen checkpoints were temporary node-local artifacts; retained manifests are not resumable Qwen checkpoints. Empty tiny-critic publication markers are omitted; the retained DCP state/metadata supports the explicit resume comparison. Every exclusion is inventoried.

## Remaining scope

This does not establish reward improvement, throughput, all model families, multi-node behavior, evaluator-placement numerical equivalence, adverse-network resilience, or replay/RNG restoration. Exact tests cover a small deterministic critic; the Qwen runs check lifecycle rather than numerical parity with another Qwen run. Active, uncapped policy ratios can still overflow for genuinely extreme eligible inputs; only excluded arithmetic and diagnostics were made safe. Cache validation checks structure/keys, not tensor-content hashes. Atomic progress replacement is not a transaction across all components.

The next substantial milestone is the canonical episode/shared-runtime migration with learning changes disabled: explicit value-owned snapshots before policy filtering/release, complete leave-one-out groups, action topology, evaluator-version warmup barriers and independent critic clocks. Multi-trace value semantics must be defined before enabling that case. Sampling replay, provenance and optional objective changes follow separately.
