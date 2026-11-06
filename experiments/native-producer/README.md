# Native producer integration

This experimental profile targets verifier `3f142cc61`, renderer `05269446`,
and pydantic-config `65b15dff` (0.4.3). The environment extra contains native
RG mix and reverse-text. The parent branch at `1801d112` and its lock retain
legacy execution. Historical configuration files remain present; a legacy
`id`/`args` entry is rejected during configuration validation rather than
reinterpreted as a native taskset. This profile is not yet a replacement for
all environments in a main-branch PR.

PrimeRL awaits the official client's completed Episode. Stream assembly,
retry discards and cancellation transport remain owned by verifier. The
consumer still selects from the original trace set, captures value inputs
before policy filtering, and requires complete privileged peer groups.
Physical token IDs remain the policy and critic coordinate system.

The Rollout adapter preserves historical branch-based total/output/input
accounting, which affects both token batching and optional input-length reward
penalties. Weighted reward records contribute once; unscored placeholders are
omitted from component metrics. The value math and optimizer precision
settings are unchanged. Generic shared-prefix text branches remain supported;
new semantic/compaction graphs, namespaced tools, multimodal reconstruction
and producer-stamped unsupported token streams fail explicitly before credit.

The optional single-turn projection remains pure and opt-in, with its
original physical trace retained outside public serialization. It keeps full
response text, rewards and provider usage while clipping only the owned
training prefix. It rejects unsupported sidecars instead of misaligning them.

Native task selection preserves explicit integer task identities, including
duplicates and gaps; only an unset idx is filled from row position. It applies
the canonical file-based system-prompt override using public `with_data`.
Pre-attached taskset views are rejected; the existing sampler still owns
selection order and lazy infinite pulls. Exact sampler restart remains the
preexisting cold-restart behavior.

PrimeRL preserves subprocess as its default agent runtime. An explicit
runtime remains honored. The old nested harness.runtime placement is migrated
once into agent.runtime. RG inline system prompts move to prompt_text;
system_prompt is now the canonical optional file override.

Renderer pooling is now the upstream elastic multiplexed pool. There is no
correct conversion from fixed pool_size to multiplex, so an explicitly
configured pool_size is rejected; the former internal default of1 is no
longer sent. This is a resource scheduling change, with no throughput claim.

The final005 identical-harness actual CPU gate passed four RG/reverse-text within-limit and
clipped cases against the frozen producer, including exact request/physical-token
and task/LOO/TETHER credit comparisons. Target-only checks cover private prefixes,
failed Episodes, real production startup and partial-stream cancellation/recovery.
Eleven independent contract checks also passed. Final paired source receipts and
compact results are recorded with this unit; these are deterministic producer
and evaluator-boundary tests, not GPU inference or learning-curve evidence.

An old inline RG system_prompt string is interpreted by the canonical schema as
a path and fails when the taskset reads it if no such file exists; it is not an
early file-existence config validator. Paths may be valid only on the execution
host, so do not add author-host existence requirements. Use prompt_text for
inline text explicitly.

A successful lock or import alone does not establish GPU backend compatibility
or token weight-version provenance. The later `experiments/weight-versions` unit admits
validated weight spans before the backend transition. Sampling-mask replay is a
separate distribution-changing feature and remains disabled/rejected; official
vLLM0.32 defaults it off. The original unit11 acceptance did not cover carriers.

Renderer parity is limited to the tested default single-turn RG/reverse cases.
A supplemental disabled-thinking multi-turn fixture intentionally changes its
physical prompt from47 to51 tokens under the new Qwen renderer, retaining the
historical empty think wrapper; the generation suffix matches. This upstream
prompt change can affect continuations even though private value ownership and
credit math are unchanged. Do not claim universal prompt parity.
