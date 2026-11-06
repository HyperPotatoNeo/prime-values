# Observed policy-version carrier

This unit carries optional engine observations from completed native episodes to
packed trainer inputs. It prepares the native producer for vLLM 0.32, which emits
weight-version spans even when sampling-mask replay is disabled. It does not
upgrade the backend, publish engine labels, compute staleness, change losses, or
supply metadata to the value function.

`MessageNode.weight_versions` remains the canonical raw evidence: opaque strings
and half-open intervals over the node's sampled completion tokens. Admission
rejects malformed or overlapping intervals. Empty intervals are valid and label
no token. The single-turn projection clips these intervals by retained sampled
count while leaving the original physical trace untouched.

The optional `token_policy_versions` transport field has a narrower meaning:
comparable live-policy step IDs, encoded as little-endian int64 bytes on the full
physical input-token axis. Only canonical `str(step)` labels (`0` or a nonzero
ASCII digit followed by ASCII digits, within int64) are comparable. Missing,
`default`, opaque, signed, whitespace-padded, leading-zero and oversized labels
become unknown (`-1`). Their original strings remain available in the trace and
JSON records. Dispatch versions never fill missing observations. Frozen-model
samplers do not emit comparable policy IDs even if their engine uses numbers.

Completion intervals map through literal True positions in the original node
mask. Shared-prefix context copies retain their sampled labels after the
canonical training mask deduplicates credit. Newly rendered prompt nodes remain
unknown: equal text or token IDs do not establish graph identity. Future
trainable-token diagnostics must use the existing training mask and count
unknown eligible tokens in their denominator.

The existing materializer owns this mapping. Packing preserves the original
physical axis, backfills missing samples and padding with unknown, and clears
labels on dummy batches. The CPU loader exposes an optional `[1, sequence]`
int64 tensor; the training loop does not shift, shard, move or consume it. CP
ranks receive the same full-axis observations as their full input batch.

`None` represents an entirely unknown stream. Known streams cost eight bytes per
token on the wire (32 MiB for 512 sequences of 8192 tokens), plus bounded packing
copies. The new fields are trailing optional fields in the array-based schema;
compatibility is checked by decoding actual old and new packets, not by assuming
identical serialized bytes. Sampling masks and other new producer credit streams
remain explicitly unsupported.

Validation is deliberately CPU-only: independent coordinate/projection controls
and an actual native HTTP/worker/message-pack/sink/packing boundary gate. Those
checks establish carrier alignment and preservation of existing credit outputs;
they cannot establish real engine label attribution across a weight reload.
That producer proof and staleness reporting belong to the backend unit.

Accepted final002 CPU evidence: 31 focused cases passed (11 new coordinate
controls and 20 existing sink regressions). The identical wire harness completed
seven target cohorts and two frozen controls, with 12 exact task/LOO context
comparisons covering absent, numeric/opaque mixed, default-only, clipped and
frozen-source observations. Literal HTTP requests, all preexisting sample/packed
fields, evaluator inputs, value credit and copied publisher payloads matched.
Actual old/new packet decoding, missing-field defaults, numeric payloads, malformed
codecs, truncation, padding, dummy batches and CPU tensor alignment were checked.
No real backend reload or model training step was run for this observation unit.
Compact receipts belong in `results/`; full raw artifacts are cached separately.
