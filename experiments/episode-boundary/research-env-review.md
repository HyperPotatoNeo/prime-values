# Research environment dependency source audit

Reviewed November 5, 2026, for the canonical Episode integration. This is a source audit, not an all-environment execution or numerical-equivalence claim.

| Role | Exact revision |
| --- | --- |
| Existing research-environments | `3e3b5ac3d75aa30f783d5b0c71e39b034cadcec2` |
| Recommended published API-port revision | `304046932a7c1f9900f82f2007514ef6f357f5b4` |
| New verifier boundary | `192fd8ef0d2fdcefc8566b0ec793d73631e7b51a` |
| Later upstream research pin, deliberately not selected | `e40d32690f74f4cdefff9dec4cdb654d63a59170` |

The selected revision moves per-task behavior from Taskset to Task, records task data under `TraceTask.data`, and replaces `load_tasks` with `load`. Task-specific settings move under `taskset.task`; custom settings such as `math_verify_timeout` must follow that path. It requires Python >=3.11 and verifiers >=0.1.15.dev443.

It precedes shared scoring helpers (`2de44bf71c495c2d15045912bdb71c9567569e80`) and the default-enabled math reference-judge fallback (`30db0678abee5eddfbef50284137ee85b153239f`). Merely disabling the later judge would not establish equivalence to the earlier isolated verifier scripts.

## Reward method comparison

For changed Python files under `environments/`, parsed old/new source with Python AST and matched methods decorated with `vf.reward`. Normalization removed the old `task` argument and annotations, replaced `task` references with `self.data`, and removed that argument from calls to migrated methods. No arithmetic, conditions, return values, or decorator weights were normalized. **36 of 46 reward methods are structurally identical** under that mechanical normalization.

The other ten were reviewed directly:

- BrowseComp, GPQA, I3 Science, Oolong Real, Oolong Synth, SimpleQA, and SimpleQA Verified: existing judge construction moves from taskset initialization to the reward call. Conditions, prompts, argument values, scoring rules, and explicit judge defaults are unchanged.
- Wideseek: the same lifetime change for its existing cell and answer judges; reward computation is unchanged.
- MultiSWE: `calculate_reward` becomes a free function with an unchanged body.
- Wikispeedia: unused `runtime` argument removed; target-reached predicate unchanged.

This does not prove identical network scheduling, judge connection lifetime, remote responses, inherited verifier behavior, or every lifecycle hook. GPQA/I3 Science already had judge fallbacks; no new math-env/AIME fallback is introduced by the selected revision. Math-env/AIME/MATH500 wrappers retain prompt text, dataset selection, script arguments, failure handling, score parsing, and default timeout values.

## Verifier payload identities

Compared the complete path-to-Git-blob maps for every `environments/**/verify.py` at both research revisions: **all 14 paths and blobs are identical**, with no additions or deletions. Each name below expands to `environments/<name>/<name>/verify.py`.

| Environments | Identical Git blob |
| --- | --- |
| aime24_v1, aime25_v1, aime26_v1, apex_shortlist_v1, math_env_v1 | `53c3359ee197611314f605a7d1861f163f592b6f` |
| arxivmath_v1, math500_v1, mmlu_pro_v1 | `31d456887e01b902cce74a420b3e5fd42065abf7` |
| i3_code_v1 | `96a5197dafadef7fb062b3ae4cd13db5bd43fdca` |
| i3_science_v1 | `2c697c93cfd8110307b332782ffb39518e3f76bd` |
| livecodebench_v1 | `bff8707b821d0e2a03110a5f5bb7a6eb4e113354` |
| longcot_mini_v1, longcot_v1 | `e92aeb7839046d12100467546498ff3e05cfe868` |
| prolog_v1 | `1a806cccf4944a4346a4e4d2e68c63187ede1ac7` |

## Explicit qualifications

Ancestor `208c230ff0a066c918db14648d81dd96bb288314` removes persona sentences from Forth, LiveCodeBench, Patterned Needle in Haystack, and Wideseek, and rewrites Prolog's persona sentence as a direct task instruction. These prompt changes can affect training. Searches of current `examples/`, `experiments/`, and `configs/` found no invocations of those five environments; a wider authored-code search found only optional dependency/source entries. Therefore no existing experiment needs an override, and no prompt patch or new runtime fork is introduced. Do not claim all native prompts are unchanged.

Ancestor `d95f4bdd26981237e8cd65063082c0270edcbeca` adds Terminal Bench's `use_prime_registry=False` option. The selected revision still overrides the obsolete `load_tasks`, so its new registry option is not supported correctly. The later `504bd26206f18e1f2e6c7722988b2622734996ac` renames that override to `load`, but still accesses old Task fields in the enabled branch; it does not completely repair the option. The default-disabled path retains prior dataset image behavior, and no current experiment invokes this environment. Do not claim support for that newly added registry option.

Recommendation: use the published API-port pin, retain existing legacy environment pins, exercise the actual native math producer/scorer with the new verifier, and keep these limitations explicit. Source parity complements the value-credit and gradient comparisons; it does not replace them.
