# Native producer and backend roadmap

This is the current roadmap after unit10 at `1801d112` and the CPU-accepted
unit11 native producer candidate. The original read-only audit informed these
units; it is no longer accurate to say the dependency pair is uninstalled.
Exact pins and staged gates are in [plan.json](plan.json). Unit11 final paired
receipts and publication status belong to `experiments/native-producer`.

Unit10 preserved the sequence boundary with a pure optional training projection.
The original discrepancy was real: the same651-token prompt and28-token response
retained3actions on the legacy path at cap654 versus28 natively, changing terminal
discounted credit. Projection now owns capped physical geometry before Rollout
consumers while retaining full response text, reward, raw usage and original
trace provenance. Native defaults remain unchanged.

Unit11 adapts verifier `3f142cc61` and renderer `05269446`, plus pydantic-config
`65b15dff`/0.4.3 and the required compatible SDK closure. Actual resolution
retains the previous numerical/GPU package versions. The supported profile is
native RG and reverse-text; complete Episode assembly stays upstream-owned,
physical tokens stay authoritative, and the existing value math stays unchanged.
Old branch-based counters are explicit because batching and input-length reward
penalties depend on them. Weighted rewards and private task prefixes survive the
new schema. Shared-prefix plain branches work; new semantic/compaction shapes
remain rejected before credit. CPU workers and fixed evaluator boundaries do not
prove the upgraded full GPU process stack.

Unit12 implements the observation-only weight-version carrier prerequisite.
Final002 actual CPU boundary acceptance passed, including 12 exact task/LOO
context comparisons and 31 focused checks. Raw opaque
spans remain canonical, while comparable live-policy step IDs map through original
sampled positions, projection, shared context and the existing materializer and
packer. Unknown stays unknown. No objective or value geometry changes.

Official vLLM0.32 source confirms that version spans are emitted with the initial
`default` label, whereas `return_sampling_mask` defaults to false. Sampling-mask
replay changes the training distribution and is not part of this carrier unit;
it remains explicitly rejected and disabled. Carrier admission must land before
or with the backend transition.

The later backend unit uses upstream `f8016025` as the vLLM0.32 / torch2.13 /
transformers5.19 reference. Review launch, token-serving, sampling defaults,
filesystem/NCCL reload and existing patches deliberately. Preserve FP32
optimization/reduction and the objective. One actual supported Qwen actor/value
fresh-and-resume acceptance is appropriate; kernel equality and performance are
measurements, not assumptions.

After the carrier exists and the engine actually provides reliable labels, add
verified startup/reload labeling and mismatch-by-age diagnostics from
`a22eeb32`/`b2df580d`. A generation spanning an actual weight update must produce
correct sampled-token spans. Unknown labels stay unknown with coverage reported;
dispatch age never substitutes for exact token provenance. Stale-token masking,
replay changes and alternative objectives remain separate.

Legacy examples remain on the frozen parent stack. The experimental native
profile fails clearly on legacy configuration before resource allocation, but
that alone is not sufficient for a main-branch replacement: account for all
legacy/MM compatibility in the migration backlog. A separate pinned legacy
worker with an explicit protocol adapter is an option only if simultaneous
execution is needed; do not embed version conditionals throughout value code.
Math migration still requires deliberate dataset/prompt/scorer compatibility.

Renderer parity is limited to the tested default single-turn RG/reverse cases.
A supplemental disabled-thinking multi-turn fixture intentionally changes its
physical prompt from47 to51 tokens under the new Qwen renderer, retaining the
historical empty think wrapper; the generation suffix matches. This upstream
prompt change can affect continuations even though private value ownership and
credit math are unchanged. Do not claim universal prompt parity.
