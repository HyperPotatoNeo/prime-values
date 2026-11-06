# Execution-only optimization and fresh attempts

The initial campaign exposed two avoidable overheads: cross-node NFS visibility
latency and redundant CPU processing of sampler candidates. Siddarth authorized
fixing both, tuning execution settings without changing the algorithm, and
relaunching fresh attempts. The latest concurrency allowance is seven 32-GPU runs
(224 H200s), with the remaining three admitted automatically as slots free up.
Original output directories and W&B runs remain historical attempts.

## Changes

- Missing batch paths refresh their parent directory entries, recursively when a
  new step directory is also missing. Completed paths retain the stat fast path.
  No shared mount options or infrastructure permissions change.
- Score-centering response assembly constructs chosen-token OpenAI fields only.
  A request-scoped ContextVar isolates concurrent regular and compact responses;
  the raw vLLM output still contains all 128 full-normalized candidates and is
  encoded into the identical compact wire payload.
- Nonstreaming token requests with no string stop and no prompt-logprob text skip
  candidate detokenization. Nonstreaming output is final-only. Streaming remains
  delta mode; string stops and prompt-logprob text retain detokenization. These
  affect output bookkeeping, not logits or sampling probabilities.
- Trainer activation offloading is disabled. Eight shards per replica, two
  replicas, activation checkpointing, FA2, BF16 compute and FP32 optimization and
  reduction remain. JSON child configs preserve explicit null values.
- Each node requests 64 CPU cores instead of 32. Inference retains 16 independent
  TP1 engines, 512 in-flight rollouts, and unchanged dispatch/off-policy limits.

`preflight_fast.py` permits only activation-offload removal, attempt identifiers
and tags in resolved configs. All original algorithm settings remain unchanged:
2000 steps, batch 256, group 8, context 8192, ancestral temperature 1, learning rate
1e-6, TIS cap 2, SC top128, EMA decay .95 and betas .003/.01/.03.

## Measured evidence before final pipeline validation

| Measurement | Before | After |
| --- | ---: | ---: |
| Actual 8192-token top128 response conversion | 10.02 s | 0.284 s |
| Live control step after NFS refresh | about 100 s | 38–44 s |
| Live control batch wait | about 60 s | 2–3 s |
| Eight-GPU trainer steady update benchmark | about 29 s | 16.5 s |
| 32 simultaneous fixed-2048-token top128 requests | 61.37 s | 17.01 s |

The response conversion returned exactly equal fields apart from the existing
random response ID. Six actual cross-node missing-file probes became visible in
4–5 ms with the new function. 32 isolated fixed-prefix inference comparisons had
identical sampled tokens, head IDs and head probabilities. Long concurrent seeded
generations were **not** bitwise identical between output modes; timings change
batch scheduling and autoregressive divergence invalidates comparing later
logprobs at different histories. The fixed-prefix check addresses probability
preservation; this does not promise identical asynchronous research trajectories.
The 2048-token ignore-EOS probe is only a throughput benchmark, not a production
configuration.

Trainer benchmarks use deterministic full-8192-token synthetic data, batch128 on
8 GPUs (16 sequences/rank, matching production batch256 on16GPUs), four updates
and the EMA .01 arm. No-offload took16.5s and24GiB/GPU. Removing recomputation
reduced this to15.3s but44GiB; four shards15.4s/57GiB and unsharded15.4s/123GiB
provided no useful additional gain. Keep checkpointing and eight shards. These
are bounded execution comparisons, not learning-quality results.

44 focused/regression/algorithm tests pass. The first full-size SC smoke with
response and NFS fixes completed two updates with finite gradients and mismatch
mean about .0005 before being stopped for the additional token-output optimization.
The final five-step pipeline check and fresh launch status are recorded in the
private research-cache performance record. Component speedups are not multiplied
or advertised as an end-to-end speedup.

Runtime: `/data/siddarth-sc-ema-fast`. Original runtime and artifacts are preserved.
`prepare_fast.py` generates new configs only; the seven-slot queue submits jobs.
Source/config hashes, timing reports, trajectory audits and resource receipts
belong in the private cache; full checkpoints/rollouts stay on the research PVC.

## Final validation and launch

The final five-step SC smoke completed naturally with all component completion
flags. Steps2–5 took35.68,38.12,35.58,37.07seconds (mean36.61). Forward/backward was
13.3–14.2seconds; batch wait6–8seconds; publication12.8–15.4seconds. Mean mismatch
was.000485–.000505, largest sampled token mismatch10.01, with finite loss and
gradients. The trajectory audit rescored516 saved answers and checked3,402,821
sampled tokens in two retained batches. Chosen-token logprobs match the compact
head whenever the chosen token belongs to it. Countdown and cryptarithm examples
were also checked manually. This is throughput/correctness evidence, not a reward
improvement claim or proof of long-run stability under seven-way contention.

The relaunch uses the manifest's current attempt mapping. Initial Service-quota
recovery interrupted control22 andSC11 during startup, before training; they were
replaced with `-fast2` attempts. Their `-fast1` artifacts and resource receipts are
preserved. The queue now creates Services before Jobs and repairs partial
admission using Services only; it never reapplies a live Job's original
`suspend: true`. Seven live research Jobs request224GPUs, with three EMAseed22
experiments waiting. Original research and performance-validation GPU Jobs are
released; a temporary CPU reader may be present during startup verification.
