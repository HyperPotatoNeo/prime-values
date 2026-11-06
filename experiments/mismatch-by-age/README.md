# Mismatch by sampled weight age

This observation-only extension finishes the useful age-bucket diagnostic from upstream `b2df580d19957e74580ca3d352d76d433cbd5e74`. It identifies which sampled-policy age groups contribute to inference mismatch before considering any changes to training or concurrency.

The existing stable, detached `mismatch_kl` tensor and its exact RL/reference-KL eligibility mask feed eight integer age bins: `0`, `1`, `2-3`, `4-7`, `8-15`, `16-31`, `32-63`, and `64+`. Unknown labels and future labels have separate populations. Every population reports count and, when nonempty, mean, maximum, and the fraction reaching the existing diagnostic ceiling. The saturation fraction makes that ceiling visible; it does not cap the policy objective or reinterpret the reported mean as an uncapped statistic. Valid NaNs propagate to the population mean and maximum; excluded NaNs do not enter any statistic.

The pre-update reference remains `step - 1` for single-run, non-LoRA training. Without a comparable reference, only unknown mismatch and the existing coverage metrics are reported; known tokens are not assigned invented ages. Unknown/future labels never become age zero. Integer comparisons preserve int64 label boundaries. CP rank zero owns each gathered full-sequence population, while all ranks participate in reductions. No transport, admission, mask, reference, numerical default, loss, or value-function behavior changes.

Only `trainer/rl/metrics.py` and one caller argument in `trainer/rl/train.py` change in production. The existing streaming reducer retains bounded summaries, not token histories. This adds fixed telemetry work; no throughput or synchronization improvement is claimed.

## Verification

Three existing CPU tests were extended, not replaced by helper-derived expectations. They passed with source-root guards for metrics, reducer, actual policy loss, canonical materializer, and packer:

- Literal observations cover every bin edge, int64 extremes, first/resumed references, unavailable references, empty/unknown/future populations, saturation and valid/excluded NaNs.
- Real Episode materialization, projection, shared-prefix deduplication, packing, component masks, padding and a dummy batch preserve the independently enumerated nine-token population. Analytic token-specific mismatch values check the resulting groups.
- The actual production policy loss, full gradients, AdamW parameters and moments remain exactly equal with reporting enabled versus disabled on a small CPU model.

The existing two-rank Gloo probe also passed five cases: unequal populations, an empty rank with unknown-only observations, CP-replicated input with one observation owner, and valid NaN on either rank with an excluded-NaN and infinity control. These test real distributed aggregation, not a GPU context-parallel model forward.

No new GPU model acceptance run was performed for this detached aggregation-only change. The preceding unit's actual actor-state equivalence is historical evidence, not a new test of these bins. Cached real rollout spans have inference logprobs but no paired trainer logprobs, so they are not presented as measured mismatch-by-age data. The CPU packed cohort uses explicitly analytic mismatch values. Results and source hashes are in `results/`; the broader dependency/backend limitations remain those of the parent experimental branch.
