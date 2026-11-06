# Bounded implementation validation — November 5, 2026

The previous `pv-rg-mix-100-001` experiment was stopped at Siddarth's request. Its last logged completed policy update was 81/100 at 2026-11-06 01:12:57 UTC. The Job and ten experiment Services were deleted around 17:18 PST; persistent artifacts remain at `prime-values-shared:/data/outputs/rg-mix-qwen4b-100-001`. This is a stopped experiment, not a completed 100-step result.

No proposed 2,000-step main run or hyperparameter pilot was launched. The validation Job has been deleted; final pod-release verification is recorded in the resource receipt. Validation used one eight-H200 reservation, `pv-sc-validate-001`, with a two-hour deadline. Pipeline checks used one inference GPU and two trainer GPUs; the synthetic hybrid check briefly used all eight. Resource receipts distinguish reservation time from model work.

## Numerical and integration checks

- **87 tests passed** on the frozen source: full-vocabulary score-centering gradient oracle; proportional-tail oracle; on-policy zero correction; off-policy EMA KL value/gradient against exact categorical KL; masked/weighted loss and gradients for all three arms; candidate head values and input/weight gradients against dense softmax across vocabulary tiles; injected Qwen model forward/backward; payload/packing; EMA clock/state continuation; broadcast acknowledgement and transport retention; existing loss/packing/trajectory regressions.
- **536 runtime source files/hashes matched** the local source manifest.
- **Score centering:** real Qwen3-4B, full 8K budget, batch 16/group 8, two optimizer updates; fresh-process checkpoint 2 → update 3. Both stages exit 0. Fresh 13m 36s; resume 6m 45s, including startup/checkpoints/teardown. These timings are not main-run throughput.
- **EMA KL:** same bounded model/task settings; two updates and checkpoint 2 → update 3. Both exit 0. Fresh 6m 35s; resume 4m 50s. EMA publication counter resumes at 2 and advances to 3. Tiny-model checkpoint tests additionally compare every restored parameter and the next EMA update exactly.
- **Hybrid sharding:** real pinned Qwen4B plus frozen EMA, 8 GPUs, two replicas × four shards, 8K synthetic batches, two updates, exit 0 in 64.7s. Peak memory 28.5 GiB/GPU. This is one node, not a two-node network or four-node end-to-end benchmark.
- All 27 generated component configurations parse through the actual CLIs and round-trip unchanged. Cluster server admission dry-run succeeds for all nine prepared Jobs and 36 Services; these resources were not applied.
- Frozen validation 256 and test 256 prompts/answer IDs pass explicit dataset preflight; no cross-split prompt overlap. The first independent-seed validation attempt contained overlap and was rejected; deterministic disjoint selection is recorded.

## Recorded data and diagnostics

| Check | Score centering | EMA KL |
| --- | ---: | ---: |
| Trained rollouts audited | 48 | 48 |
| Sampled tokens checked | 235,032 | 197,715 |
| Saved trajectories independently rescored | 55/55 agree | 60/60 agree |
| Mean trainer/sampler mismatch across steps | .000444–.000503 | .000460–.000525 |
| Largest token mismatch | .4687 | .4111 |
| Largest sampled importance ratio | 1.9581 | 2.0626 (policy-gradient cap2) |

All audited token streams are aligned and finite. In score-centering data, 235,027 of 235,032 sampled actions lie in the recorded top 128; those entries match the separately recorded sampled logprob within 1e-6. The five outside-head actions retain their own logprob. At update 2, mean omitted sampler mass is about 1.49e-5 (epsilon floor 1e-6); the maximum is .00982.

EMA KL mean is .000587 at update 2 and .000667 after resume/update 3; maximum sampled KL at update 3 is .9473. Update3 gradient norm is .3236; no nonfinite failure was observed. Negative policy-gradient surrogate losses are valid and are not themselves a failure.

Manual trajectory inspection included correct arithmetic using all seven numbers, a complete cryptarithm mapping, an incomplete mapping appropriately receiving .01, and a correct ARC shift. Exact rescoring validates consistency with the existing RG scorer, not independent certification of every environment's reward design. Short smoke rewards are not comparable learning results: asynchronous arrival/filter settings and small task samples differ. No improvement claim is made.

## Preserved failures and limits

1. A 256-generation-token probe produced no usable advantage variation and hit the empty-batch guard. It was replaced with the requested full 8K budget.
2. The first full-budget score-centering probe found that the injected model-forward wrapper did not route candidate IDs. The wrapper was fixed; a real injected-model test and successful fresh/resume pipeline cover it.
3. The first validation split overlapped training/test despite a different seed; it was rejected and replaced with deterministic disjoint selection.
4. Kubernetes control requests intermittently returned 401 despite a valid reusable session. Bounded retries and an established authenticated exec stream worked; no infrastructure permission or policy changes were made.

The smoke's first two SC updates used the default zero-advantage post-filter; the resumed SC update and all EMA smoke updates used the final monitor-only filters. Main configurations use monitor-only filters for every arm. Smoke runs establish execution, not an unbiased method comparison.

The paper's top-k proportional-tail approximation is implemented, not a full-vocabulary exact centering estimator. The EMA regularizer is an importance-corrected sampled reverse KL at rollout prefixes, with potentially high variance under strong staleness. Main-run actual lag, four-node throughput, long-horizon stability and relative learning performance remain empirical questions. The final evaluation runner passes dataset preflight but has not generated a complete held-out evaluation for this comparison.

Detailed configs, receipts, audit JSON and diagnostic logs are under `prepared/` and `results/`; `evidence-manifest.json` locates the raw archived trajectories, token exports, payloads and logs. Full checkpoint tensors are intentionally excluded from the retained validation archive.

Runtime evidence is published in the [private Resolution cache](https://github.com/resolutionorg/siddarth-research-cache/tree/main/projects/prime-values/records/2026-11-05-score-centering-ema). Repository access and cluster artifact access are separate.
