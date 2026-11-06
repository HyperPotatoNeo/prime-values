# Canonical Episode integration

Completed against upstream Prime RL landmark `cb74ae2b`, preserving the value-owned trajectory and credit boundary from `4b8ca528`. This is a selective synchronization unit, not all of latest upstream.

`PYTHONPATH=src uv run python experiments/episode-boundary/producer_smoke.py --output /path/to/evidence/producer.json`

The smoke runs the pinned native `EnvServer` and maintained `LegacyEnvServer` over their real ZMQ protocol. The native null harness runs in a subprocess. A local HTTP model fixture returns a fixed answer; no credentials or GPU are needed. This validates canonical episodes, task-data extras, config serialization, legacy runtime limits and joint group scoring. It does not claim tokenizer, inference-kernel or learning coverage; the RL safety pipeline supplies those checks with a real model.

The native spawn case also exercises the orchestrator's multiprocessing launcher, `serve_env(config_data=...)`, server startup handshake and child shutdown. Recorded results are under `results/`, alongside real reverse-text server initialization and launcher TOML reload evidence. All seven shipped native debug configs (ten train/eval environment entries) validate and reload; this is configuration coverage, not a claim that their remote tools or datasets have all run.

The legacy group rubric returns rewards 2 and 3 only when both siblings reach joint scoring. Per-rollout scoring, missing siblings, wrong answers or failed adaptation cannot satisfy that oracle. The native task includes a privileged value prompt, checked after wire deserialization. Multiple-agent episodes remain explicitly unsupported at the rollout adapter.

Native configuration uses `env.taskset`, `env.agent.harness`, agent limits under `env.agent`, and interception pooling under `env.interception`. The environment entry's `pool` config still sizes server worker processes. A top-level `id` selects the upstream legacy bridge. Existing flat single-agent configs translate with a warning; mixing flat and nested options fails rather than choosing precedence.

Dependency scope: verifier `192fd8ef`, renderers `bdb96b0c`, pydantic-config `4f5ae373`, and research-environments `304046932`. Five removed legacy example packages retain their exact original definitions through git sources at verifier `9c6b155`; they use the current verifier runtime. The lock preserves all nine numeric package/source entries for Torch, vision/audio, Transformers, vLLM, Flash Attention and NumPy. The required sandbox SDK advances only from 0.2.27 to 0.2.31.

Color-codeword additionally pins its environment package to `6c64ce6`, which has the task-centric API while preserving the finite 1,000-example configuration. The newer bundled package changes it to an infinite taskset; deleting its old limit would change behavior and exceed this unit's supported producer scope. The real pinned server loads exactly 1,000 tasks. All 180 tests in `tests/unit/test_configs.py` pass with actual environment packages available, with no plugin mocks or skipped failures.

A clean Git wheel build derives renderer version `0.1.9.dev1` from `bdb96b0c` and its ancestor release tag. The deployment uses this clean-built wheel; stale archived editable metadata reporting `0.1.8.dev54` is not the actual build version. `results/renderer-build.json` records the wheel hash and build provenance; no version override or renderer source upgrade was needed.

The research-environments pin deliberately precedes later math scoring and default LLM fallback changes. Its math/AIME verifier scripts remain identical to the baseline. It includes upstream removal of persona introductions for five unused native tasksets (forth-lang, livecodebench, patterned-needle-in-haystack, prolog, wideseek); broad prompt parity for those tasksets is not claimed. Terminal Bench's newly introduced registry option is also outside this unit's support claim and remains disabled by default. Current math, reverse-text and legacy value experiments do not use those changes.

## Preservation evidence

The focused CPU suite passes 168 checks, including real `TrainSink` and GRPO control flow with deterministic evaluator/renderer/publication boundaries. The checks cover arrival-time critic versions, analytical advantages and targets, publication before policy filtering, refilling, and complete versus cancelled leave-one-out groups. They do not replace the real producer or model checks.

Eight frozen-versus-current cases exactly match finalized credit, evaluator requests, TETHER controller continuation, actor and critic losses, and every input-logit gradient. These compare the production paths to the frozen safety revision `f6b9a40c`; they are not a claim of full transformer-parameter or stochastic training equivalence. Raw outputs and the comparison are under `results/credit/`.

Four prompt cases produce twelve byte-identical leave-one-out prompts against `4b8ca528`, covering shared prompt ancestry, tools, reasoning, stop reasons, call ordering and group permutation. `compare_prompts.py` and `prompt-parity.json` record this narrower contract. Finish reasons now come from committed `ModelCall.node` references; unsuccessful calls without a node cannot change a peer's prompt.

`results/cpu-summary.json` records the actual focused/config test counts and artifact hashes. The archived CPU environment uses editable dependency paths relative to its original scratch location; reconstruct it against the recorded source and dependency revisions. It is an evidence snapshot, not a portable install command from this directory.

## Real-model lifecycle validation

All five stages passed using Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554` on one eight-H200 node. Each stage ended naturally with exit code 0; no continuation workaround supplied missing exit evidence.

| Placement | Fresh updates | Restart updates | Restored policy counters |
| --- | --- | --- | --- |
| Dedicated value evaluator | Policy 1–3; critic 1–4 | Policy 4; critic 5 | 2,920 tokens / 6 packed samples, then 3,904 / 8 |
| Trainer-hosted evaluator | Policy 1–3; critic 1–6 | Policy 4; critic 7–9 | 2,802 tokens / 6 packed samples, then 3,752 / 8 |
| Value disabled | Policy 1 | Not part of this check | 882 tokens / 2 packed samples |

Both value-enabled runs verified warmup, live evaluator versions, finite losses and policy gradient norms, intermediate checkpoint counters, fresh-process restoration and weight publication. The value-disabled run verified a finite actual policy update and no value process/checkpoint. Different critic update counts reflect asynchronous scheduling; these are not placement-equivalence or reward-improvement results.

The resolved configs retain ancestral sampling, group size 2, rollout/value batch size 8 and the unchanged optimization/reduction precision. These bounded smoke runs use offline logging. The existing Torch/CUDA/Transformers/vLLM stack stayed fixed. Its pre-existing cuDNN package-metadata discrepancy is recorded in `results/gpu/validation-environment.json`; the repository's locked override was preserved.

Root review verified all 265 archived file hashes and all 319 runtime code/test/config fingerprints against the final checkout. Initial source verification covered 2,310 regular files plus two symlinks, and 24 renderer wheel Python files; regular file hashes were rechecked after training. Dedicated restart includes one narrowly classified expected inference-engine shutdown event. This is disclosed in its result, not treated as a training failure or ignored generically.

Compact results are under `results/gpu/`. The full log/config evidence archive is 435,807 bytes, SHA-256 `927e19b27291e360c48c071e8a03a65959b24d4b129d84618a55335c3af90164`, at `prime-values-shared:/data/episode-sync-001/evidence.tar.gz`. Full Qwen optimizer/model checkpoints are excluded and reside only on disposable node-local storage; this archive is not a resumable model checkpoint. The node is temporarily retained for the next separately recorded synchronization unit.

Intermittent cluster inspection authentication failures were operational monitoring problems, separate from the completed training outcomes. A persistent authenticated connection recovered the final evidence. No administrator action or infrastructure-policy change was required.

The renderer scope review confirms Qwen3 and Llama renderer source files are unchanged at the new pin; existing parsing functions change only for Laguna. New model renderers and Laguna template changes are outside broad parity claims. Numerical tests, source review and real lifecycle checks complement one another; they do not prove every model/environment, distributed topology, sampler continuation or long-run learning outcome.
