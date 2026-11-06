# Native RG environment acceptance

Adds `rg-mix-v1` alongside `rg-mix-env` on the existing verifier b13ba60d, renderer bdb96b0c and vLLM 0.24 stack. Existing experiment routes remain available. The [package guide](../../packages/rg-mix-env/README.md) shows native configuration and cached-data split semantics. The scorer is pinned to the observed `reasoning-gym==0.1.25`; all 460 existing non-root lock entries remain unchanged.

The frozen reference in `frozen-legacy/` preserves the observed original RG source. It was untracked in the original checkout, so its [source receipt](results/source-receipt.json) records file hashes rather than claiming that checkout's Git revision pinned its bytes. The research dataset remains external: use its four recorded hashes, not a regenerated dataset with the same seed.

## Accepted behavior

- All 10,256 saved rows preserve ordered train/eval selection, split-local task indices, global scorer indices, exact system/question prompts and private scoring payloads.
- Eighteen parser cases preserve first/last-tag and missing-answer behavior. One hundred actual scorer calls across all five variants preserve positive, zero and partial rewards, raw named metrics, zero-weight formatting and scoring order independence.
- Eight selected cases pass through the frozen legacy server and two native worker processes. Actual Qwen renderer requests, nonempty policy token/mask/logprob arrays, completion flags and owned critic snapshots match exactly. Workers receive selected data and cannot access the dataset path.
- The actual TrainSink processes mixed rewards 0/1 in task and group leave-one-out modes. Evaluator requests, value predictions, advantages, targets, peer prompt bytes and published critic samples match across all three producers. EnvClient cancellation is followed by successful worker recovery.
- A freshly built wheel imports all five modules from installed wheel bytes. Both literal environment IDs load the saved data through actual verifier discovery.

The HTTP token generator and value evaluator are deterministic fixtures. This is CPU boundary/algorithm acceptance, not model inference, a learning curve or GPU throughput evidence. Cancellation covers the client/worker path; it does not establish new dispatcher accounting behavior. This additive package does not switch existing run configurations. Legacy `max_seq_len` clipping occurs before Episode materialization, whereas native traces retain physical tokens until downstream limits. The covered short-response comparisons do not establish overlength parity; migrating run configurations requires the next sequence-boundary unit.

Failed and superseded harness attempts are retained in the external evidence archive; [the acceptance receipt](results/acceptance-receipt.json) explains each.

## Reproduce

Use the pinned environment with the environment extra installed, the original cached dataset and the recorded local Qwen tokenizer files:

```sh
uv run --no-sync python experiments/native-rg/package_parity.py --dataset /data/rg-dataset --frozen-package experiments/native-rg/frozen-legacy --source-root . --output /tmp/rg-package-parity.json
uv run --no-sync python experiments/native-rg/acceptance.py --dataset /data/rg-dataset --tokenizer /path/to/recorded/qwen-tokenizer --frozen-legacy experiments/native-rg/frozen-legacy/rg_mix_env.py --output /tmp/rg-producer-acceptance
```

Output paths must be new. Compact final receipts are in `results/`. The full 51-file evidence archive (including raw three-way outputs and earlier attempts) has SHA256 `04506adf766de50bfa50f372ab9b94cdc9e712527b8b357415820bddbeeaf995`; the external research cache owns its durable location. Full dataset/tokenizer caches and checkpoints are not included here.

Changed package modules and acceptance harnesses pass Ruff. The byte-preserved dataset generator retains two pre-existing import lint findings; it was deliberately excluded from this changed-code lint check.
