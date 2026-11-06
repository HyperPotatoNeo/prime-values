# Explicit packing boundaries

This unit adapts the boundary intent of upstream `8dcf55acbcc01b4792e86b61ce411d2d803e8a7c` while preserving current padding, SFT and backend behavior. It was integrated after the task-ownership unit and independent frozen-first GPU acceptance.

Existing CPU `sequence_lengths` supply one `PackedSequenceMetadata` object: owned lengths and offsets, a Python maximum, and an int32 cumulative-offset tensor on the model device. Construction validates positive integer lengths, total token count and int32 capacity before device allocation. Context-parallel local boundaries are intersections of CPU intervals; rotary positions remain unchanged. Global boundaries still go to ring/Ulysses attention. No transport schema, materialization, padding or loss change is included.

The policy entrypoint and value trainer/shared inference path use this interface only when the model explicitly supports it and uses FlashAttention. Enabled families are Qwen3, Qwen3 MoE, Llama, GLM4 MoE, AFMoE, MiniMax M2 and Laguna. Both the outer causal-LM forward and inner backbone accept the optional metadata. ValueFunctionModel retains the same head, alignment and outputs and forwards metadata only to a supported backbone. Dedicated and trainer-colocated evaluation share the changed inference helper.

Absent metadata retains the existing position-based path, including direct model and SFT callers. Eager/SDPA, ordinary HF fallback, GPT-OSS, GLM-DSA, Nemotron and Qwen3.5 paths retain their existing behavior. This does not broaden critic CP/VLM support or claim correct packed hybrid CP: existing recurrent CP code has separate document-boundary limitations. Actor padding remains a separate masked attention segment; value dummy ranks retain one masked token and are omitted from returned samples.

Validation is intentionally layered:

- `tests/unit/utils/test_packed_sequence_metadata.py` uses independently enumerated token labels to check boundaries and CP fragments, including a shard starting inside a document; invalid metadata must fail before allocation.
- `model_oracle.py` runs real Qwen/FlashAttention actor and critic computations. Run it unchanged against the frozen source with `--mode position`, then the candidate with `--mode explicit --reference <frozen>/tensors.pt`. It compares full outputs, gradients and an actual optimizer update, with exact frozen/candidate equality for identical layouts. Its fixed packed-versus-independent tolerances must pass the frozen source first. BF16 SGD checks do not replace production FP32 optimizer/reduction validation.
- A dense two-rank CP check must compare global and sharded outputs/gradients for affected ring/Ulysses paths. Do not extend that result to hybrid recurrent CP.
- `tests/integration/value_baseline_probe.py --compiled` exercises the actual two-rank ValueTrainerRuntime, FSDP, default block compilation and FP32 parameters/Adam moments for two updates, including a dummy rank. It requires a fresh output directory and an explicit source root, verifies compiled callables and captured graphs on each rank, and compares full model/optimizer states and nonzero updates exactly. Run one unchanged shared harness against frozen and candidate sources; this bounded check complements the mathematical and serving oracles without repeating a full model-training pipeline.
- Profile boundary preparation separately from end-to-end timing before making a performance claim. External ring-attention setup may still read device metadata; the scoped profiler does not establish that the whole context-parallel setup is free of host synchronization.

Local metadata tests are CPU evidence only; the completed GPU acceptance below supplies the numerical/runtime evidence. Staging review receipts retain their original pre-GPU scope. All source and validation receipts belong under this experiment, independently of the prior Episode and task-ownership results.

For the isolated GPU validation, copy the final probe unchanged to `/data/packed-boundary-001/harness/value_baseline_probe.py`. Run the frozen source first, using the existing image environment and fresh output paths:

```sh
PACKED_SOURCE=/data/task-ownership-001/source
PYTHONPATH="$PACKED_SOURCE/src:$PACKED_SOURCE/packages/prime-rl-configs/src:$PACKED_SOURCE/deps/verifiers:$PACKED_SOURCE/deps/renderers:$PACKED_SOURCE/deps/pydantic-config/src" uv run --no-project /app/.venv/bin/torchrun --standalone --nproc-per-node=2 /data/packed-boundary-001/harness/value_baseline_probe.py --compiled --source-root "$PACKED_SOURCE" --output /data/packed-boundary-001/output/compiled-frozen

PACKED_SOURCE=/data/packed-boundary-001/candidate
PYTHONPATH="$PACKED_SOURCE/src:$PACKED_SOURCE/packages/prime-rl-configs/src:$PACKED_SOURCE/deps/verifiers:$PACKED_SOURCE/deps/renderers:$PACKED_SOURCE/deps/pydantic-config/src" uv run --no-project /app/.venv/bin/torchrun --standalone --nproc-per-node=2 /data/packed-boundary-001/harness/value_baseline_probe.py --compiled --source-root "$PACKED_SOURCE" --output /data/packed-boundary-001/output/compiled-candidate

PYTHONPATH="$PACKED_SOURCE/src:$PACKED_SOURCE/packages/prime-rl-configs/src:$PACKED_SOURCE/deps/verifiers:$PACKED_SOURCE/deps/renderers:$PACKED_SOURCE/deps/pydantic-config/src" uv run --no-project /app/.venv/bin/python /data/packed-boundary-001/harness/value_baseline_probe.py --compare-compiled /data/packed-boundary-001/output/compiled-frozen /data/packed-boundary-001/output/compiled-candidate
```

The compiled probe retains the production runtime's kernel settings. If exact comparison fails, establish frozen-run repeatability before attributing the difference to the candidate; do not relax tolerances after seeing candidate results. The standalone actor/critic and CP probes explicitly request deterministic kernels for their separate exact comparisons.

## Completed acceptance

The 16 CPU boundary checks passed. Actual GPU execution then established exact frozen/candidate equality for 220 dense/MoE/HF model tensors and 77 tensors for each of two-rank ring and Ulysses context parallelism. The initialized frozen model also repeated exactly. These cover full outputs, gradients, real SGD updates, private/context tokens, masked padding, dummy value ranks and serving reassembly; compare the per-case receipts under `results/`.

The real two-rank `ValueTrainerRuntime` compiled two decoder blocks on each rank and captured two graphs. FP32 optimization/reduction and Adam moments were preserved. Initial state and both updates matched exactly, including optimizer states and dummy-rank behavior. The frozen compiled control was reused from the preceding attempt because its standalone harness and guarded sources were byte-identical; `results/compiled-control-reuse.json` records that explicitly. No additional large-model pipeline or reward experiment is claimed for this unit.

The parent independently checked all 120 archive member hashes, 329 executable/config/test source hashes against the integration tree, all 23 changed files against the actual runtime snapshot, and the paired saved tensors. Boundary profiling over ten repetitions reduced `nonzero` from 10 to 0 and `item`/scalar reads from 20 to 0 in the measured helper. Host-to-device uploads remain, and the external ring helper still performs scalar reads. This is evidence about boundary preparation, not an end-to-end throughput measurement.

Two failed harness attempts remain in the archive. First, evidence writing assumed the namespace package had a file path; the fix uses a concrete module. Second, direct tiny-MoE construction left custom expert matrices uninitialized; the fixture now calls the real initializer, verifies distinct finite expert weights and records the initial state. The routed-expert positive control and numerical tolerances were retained. The independent reviewer also found a standalone patch serialization issue in the XML result's missing-final-newline marker; the final patch round-trips all intended bytes. Runtime source was already correct, so no numerical rerun was needed for that packaging correction.

Raw evidence, including tiny-model tensor snapshots and failed attempts: `prime-values-shared:/data/packed-boundary-001/evidence.tar.gz`, 50,050,930 bytes, SHA256 `d75d0f4d850c9c1934b5ce12bffec7d8fceb907b6080b2925f14d29416abe6cb`. Compact receipts and provenance are cached in the Resolution research cache. Full Qwen training checkpoints are not included. The retained node is one shared allocation across validation phases; process/activity observations are not utilization measurements.
