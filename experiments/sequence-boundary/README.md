# Explicit training-token projection

Native environments retain their complete token geometry by default. An optional
`training_projection` on a training environment selects a prefix for policy and
value training after the environment has generated and scored the full response.
It does not change inference sampling, completion text, reward, model call records
or provider usage counters.

```toml
[[orchestrator.train.env]]
name = "reverse-text"
training_projection = { type = "single_turn" }
env.taskset = { id = "reverse-text-v1" }
env.agent.harness = { id = "null", runtime = { type = "subprocess" } }
```

An omitted `max_seq_len` resolves to the actor's `seq_len` and is saved explicitly
in resolved configuration. Set a positive `max_seq_len` in the projection table
to choose a different training prefix. Evaluation and legacy environments do not
accept this setting; the legacy producer already applies its existing token cap.

The supported input is one successful model call and one linear text branch,
with no tools, multimodal data, expert-routing or kept-token sidecars. Unsupported
successful inputs fail explicitly rather than silently changing their geometry.
Failed episodes retain their existing failure path. No historical training
configuration is changed by adding this option.

The adapter copies the trace and clips aligned token, sampled-mask, content-mask
and logprob arrays before policy/value consumers see it. Full messages and empty
suffix nodes remain, so peer context still describes the complete completion.
If clipping occurs, `max_total_tokens` is a projection-induced canonical stop
marker. It reports effective training truncation; it does not claim that the
model stopped for this reason or reproduce a legacy stop-label spelling. The
`prime_rl_training_projection` receipt retains the original stop condition and
flags, physical/retained token counts, and a projected-training-trace label.
There is no inferred prompt-overflow flag; zero retained actions alone do not
distinguish an exactly full prompt from a prompt that exceeds the cap.

`Rollout.source_physical_trace` retains the original producer trace for in-memory
audit and is excluded from serialization. Opting in therefore retains both raw
and projected node payloads until the rollout is released. This is a correctness
and migration option, with no throughput or memory improvement claim.

The initial discrepancy and actual same-response CPU reproduction are recorded
separately from implementation acceptance. The tests exercise aligned clipping,
unchanged raw inputs and usage, token filters, value ownership, rejection of
unsupported shapes, and resolved configuration. The real producer/sink acceptance
uses the existing cached RG task and pinned renderer with a deterministic token
endpoint; it is not model-inference or learning-curve evidence.

Acceptance passed 37 focused checks and the actual launcher TOML-to-child-config
roundtrip with inherited and explicit limits. The final producer check used 18
identical HTTP requests across within-budget, exact-budget, completion-clipped
and prompt-overflow cases. Legacy and projected-native policy arrays, rewards,
effective truncation, task/leave-one-out credit, peer prompts and publication
outcomes matched. Full native controls retained their original behavior. A bad
tail was filtered only while visible; clipping changed terminal credit as
expected, and zero-action cases published neither policy nor critic samples.

One existing producer metadata difference remains explicit: the legacy bridge
reports missing reasoning-token usage as `0`, while native reports unknown
(`None`). Other usage fields match, and projection preserves the entire native
usage record. Compact receipts are in `results/`; the full raw paired evidence,
including failed diagnostic attempts, is retained in the research cache.

## Reproduce

With the pinned environment and original external dataset/tokenizer available:

```sh
uv run --no-sync python experiments/sequence-boundary/acceptance.py --source . --dataset /data/rg-dataset --tokenizer /path/to/recorded/qwen-tokenizer --legacy experiments/native-rg/frozen-legacy/rg_mix_env.py --output /tmp/sequence-boundary-acceptance
```

Use a new output directory. The [acceptance receipt](results/acceptance-receipt.json)
records exact settings and failed attempts. The compact 71-file evidence archive
has SHA256 `fe08f1983ab0fd28ff2157a3845688d63e9c5146f82548c1353c7503f9bb7a2c`.
The research cache retains it; the full dataset and tokenizer remain external.
