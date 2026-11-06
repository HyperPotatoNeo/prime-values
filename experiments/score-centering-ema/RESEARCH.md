# Score centering versus an EMA reference on RG mix

Status: implementation validated; user authorized the ten-run campaign on November 6 UTC. Launch and live status are recorded separately in the private research cache. See [validation evidence](VALIDATION.md).

The question is whether score centering improves sustained asynchronous RL beyond a conventional KL regularizer to a slowly moving reference. Passing the implementation checks does not answer this empirical question.

## Methods and primary sources

- Marek & Ryabinin, *Score Centering Stabilizes Off-policy Reinforcement Learning*, September 17, 2026: https://arxiv.org/html/2609.20807v1 . Implement Appendix A, Eq. 14, with a proportional-tail approximation. Original executable implementation and sweeps: https://github.com/martin-marek/score-centering .
- NeMo RL documents the same sampler-head/proportional-tail estimator and preserves full-normalized probabilities: https://docs.nvidia.com/nemo/rl/nightly/apidocs/nemo_rl/nemo_rl.algorithms.loss.score_centering.html .
- verl's current implementation limits score centering to REINFORCE with token-level importance corrections and recorded vLLM top-k: https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr.md . These restrictions motivate a separate matched research loss rather than silently attaching a centering term to the existing DPPO loss.

Write p for the current trainer distribution, q for the distribution actually used to sample each token, and A for its detached group-relative advantage. All main arms use `w = min(p/q, 2)` and the token-summed surrogate `-A * stop_gradient(w) * log(p)`, divided by the same global number of eligible RL tokens. Advantages are raw reward minus group mean, without standard-deviation normalization. There is no value function, PPO clipping, DPPO mask, or hidden existing mismatch penalty in this research loss. All rollout filters are monitor-only in these comparison configs, including zero-advantage detection. Retaining zero-advantage rollouts keeps the batch denominator matched and allows EMA KL on those samples. This is an intentional change from the default enforcing zero-advantage post-filter.

Score centering adds `A * sum_H stop_gradient(q_v*w_v - alpha*p_v) * log(p_v)`, where H is the sampler's top-k head, rho = sampler tail mass / trainer tail mass, and alpha = rho * w(1/rho). Tail masses use the paper's epsilon floor of 1e-6. The correction's coefficient is exactly 1; no separate beta is needed. k defaults to 128. This does **not** top-k truncate generation. The actual sampled token's probability is recorded separately, including when it lies outside H. All coefficients and the advantage are detached.

EMA regularization adds beta times reverse KL(p || e). e is a frozen sharded copy of the policy, initialized from the original inference weights and updated at each publication: `e <- .95*e + .05*published_policy`. The input weights match the BF16 publication stream and EMA accumulation keeps the configured optimization dtype (FP32 here). This is weight averaging, not averaging logits or probabilities. At update t, loss uses the EMA of prior publications; the freshly trained policy enters the EMA after publication for the next update. The EMA has no optimizer. Its full sharded state, decay, update count and last publication are checkpointed with the policy; resume fails if EMA state is absent or decay differs.

The KL estimator on actions drawn from q is `(p/q) * (exp(log(e)-log(p)) - (log(e)-log(p)) - 1)`. Its expectation and gradient are exactly those of reverse KL(p || e) at sampled prefixes. Its p/q factor is **not clipped or detached**; doing either changes the regularizer. It is a sampled estimator, not exact full-vocabulary KL on every prefix. Large off-policy ratios can increase variance; log the ratio distribution and KL tails. The EMA averages every published version, which may differ from averaging only versions each asynchronous server ultimately adopts.

## Fixed main-run settings

| Setting | Value |
| --- | --- |
| Model | Qwen/Qwen3-4B-Instruct-2507, revision cdbee75f17c01a7cc42f958dc650907174af0554 |
| Dataset | Existing RG mix 0.1.4, 10,000 training tasks, dataset seed 42 |
| Training | 2,000 optimizer updates; 256 admitted rollouts per batch; 8 responses per prompt (normally 32 complete groups; errors/cancellations can leave partial groups) |
| Context | 8,192 total prompt + response tokens; EOS or remaining context, no smaller generation cap |
| Sampling | T=1, top_p=1, top_k=-1 (disabled), min_p=0, presence/frequency penalties=0, repetition penalty=1 |
| Optimizer | AdamW, LR 1e-6, betas .9/.999, weight decay .01, max gradient norm 1, constant scheduler |
| Numerics | Original FP32 optimization/reduction; BF16 model compute/publication; full parameter training |
| Trainer topology | 2 nodes × 8 H200; 8-way sharding within each node and 2-way data replication; CP=1 |
| Trainer memory/performance | FlashAttention 2, existing compilation/activation checkpointing; optimizer states on GPU, default activation offloading retained; output-head chunk 1,024 tokens |
| Inference | 2 nodes × 8 independent TP=1 engines (16 replicas); generation_config=vllm to avoid model-card sampling overrides |
| Main arms | TIS baseline; TIS + score centering (k=128); TIS + EMA KL (decay .95, beta .003/.01/.03) |
| Seeds | 11, 22 paired across all five settings; configurable prompt-order and trainer seeds; distinct engine seeds derived from run seed and engine index |
| Async | Dispatcher pause threshold 8 batches, 512 in-flight rollouts, cancellation limit 20 policy changes; already-in-flight work can overshoot the scheduling threshold; record actual lag |
| Checkpoint | Every 100 updates, latest 2 full optimizer/EMA states + latest 2 weights; acknowledgement-based broadcast retirement; retain 2 consumed binary batches, gzip older trajectory JSON losslessly |
| Logging | Online W&B, swish/siddarth-prime-values, arm/seed groups and related trainer/orchestrator names |
| Concurrency | At most 3 runs / 96 GPUs, accounting for other Siddarth jobs against the ~120 GPU soft ceiling |

vLLM scheduling, asynchronous completion order and distributed kernels mean identical seeds do not promise bitwise reproducibility. The seed pairing controls initialization, task order and sampling streams as far as the execution model permits. Record actual versions, environment hashes and resolved configs with each run.

## Comparison and evaluation protocol

The authorized campaign is five settings × two paired seeds: control, top-128 score centering, and EMA beta .003/.01/.03. There are no top-32/top-64 runs or separate tuning pilots. Report all beta outcomes; do not select on the held-out test set. The frozen validation and test manifests remain available for subsequent checkpoint evaluation.

Compare the same 2,000-step budget and report wall-clock/GPU-hour cost separately, since EMA requires an extra forward pass. Report reward versus orchestrator update, unsmoothed data plus a clearly labeled trailing mean, per-seed curves, per-task-family results, held-out reward at fixed policy versions, truncation, zero-advantage group fraction, dropped/error rollouts and actual policy-age distributions. Two seeds provide limited statistical power. If actual staleness remains small, do not claim a high-staleness comparison; review a controlled delayed-sync follow-up separately.

The legacy environment adapter's in-loop evaluation uses the training dataset. Do not label that held-out. Use an explicit `get_eval_dataset()` evaluator, pinned checkpoint and ancestral settings. Evaluation must record checkpoint version and all failed/truncated examples. Development smoke runs are functional tests and cannot establish improved learning.

## Launch boundary

`prepare.py` emits configs and Kubernetes manifests; it does not submit them. Siddarth authorized launching the ten-run campaign; use the three lane manifests, not all ten individual job manifests. Kueue can unsuspend submitted jobs, so `suspend: true` is not an approval mechanism: do not apply these manifests until asked to launch. The W&B secret is referenced by name only; provision securely from the existing Keychain entry at launch. Never put credentials in Git, command arguments or logs.

## Storage and recovery

Top-128 arrays are large transient transport data. After every rank consumes a batch,
opt-in retention removes binary batches older than the latest two and losslessly
compresses the corresponding trajectory JSON. Unconsumed/future batches are never
retired. A checkpoint restart restores policy/optimizer/EMA and regenerates rollouts;
this is not bitwise replay of asynchronous task order. Full tensor payloads for all
2,000 steps are not retained. Check live disk space before each wave: reserve roughly
200 GiB per active run plus shared/model/old-experiment storage, and archive completed
wave checkpoints before starting further waves if needed. The PVC was expanded from 1 TiB to 3 TiB for this campaign, preserving earlier artifacts. Each lane stops before its next run if less than 200 GiB remains. Three fixed four-node Jobs bound the campaign to 96 GPUs; each role advances to the next independent run only after all four roles finish. Failure halts that lane instead of silently skipping a failed comparison. Every run has a 72-hour bound; lanes have a 14-day deadline.

## Reproduction and launch preparation

1. Check out the research branch and initialize its pinned submodules. Run
   `uv run --no-project python experiments/score-centering-ema/apply_dependency_patches.py`,
   then `uv sync --all-extras --frozen` in the pinned CUDA image. The source patches
   are versioned with the parent repository; the submodule base commits remain unchanged.
2. Use `prepare.py` to regenerate the ten main configurations and three lane manifests. Confirm the frozen
   source manifest, pinned model snapshot, RG data and `splits.json`.
3. `prepare_splits.py` creates the independent validation set, rejecting prompt
   overlap with train/test and keeping deterministic candidate-selection provenance.
   `evaluate.py --preflight` checks frozen IDs/hashes without generating answers.
   Full evaluation uses a dedicated endpoint and an explicit stable checkpoint,
   never the live training inference pool. No full validation/test evaluation has
   been run for this proposed comparison.
4. Before an authorized launch, refresh live GPU and disk accounting, inject the
   existing W&B Keychain credential as the named namespace Secret, and rerun server
   dry-run and source/config checks. At most three 32-GPU jobs run concurrently;
   reduce that if other active Siddarth work approaches the 120-GPU soft limit.
5. After launch, inspect early trajectories, actual policy-age distributions and
   throughput before treating the configuration as a high-asynchrony experiment.
   The four-node throughput/ETA has not yet been measured.

The independent Episode/task-ownership synchronization branch continues separately.
This research branch starts from the completed safety milestone f6b9a40c; it does
not silently incorporate that other branch's ongoing changes.

Scalar metrics are also appended to per-component JSONL files in each run’s `logs/` directory, keyed by the true component step and observation time. W&B remains online. See `preflight.py` for CLI roundtrip checks and equality of all paired-seed configuration fields except correction controls, sampler-head capture, and run identifiers/tags.
