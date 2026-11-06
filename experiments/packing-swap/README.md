# Vectorized packing swap refinement

This unit adapts upstream `87ddbcf08179a113e76cd8650511f40e2a6a0e15` in the existing trainer packer. NumPy scores each pair of rank partitions at once, preserving the exact `(maximum load, load spread)` objective and first-swap order among equally best candidates. The original scalar implementation remains the fallback for non-plain integers and values outside the safe int64 range. The bound is checked after each accepted swap; arbitrary-size Python integers and signed weights retain their existing behavior. No packing cost, padding, sample order contract, component normalization, value logic or configuration changes.

## Validation

`verify.py` loads the actual pure helper definitions from the frozen and candidate files. All 130 direct cases and 12 public `balanced_partition` cases match exact ordered partitions, including ties, signed/empty inputs, huge integers, large totals with safe relative loads, explicit fallback and input nonmutation. The unchanged frozen implementation is the independent oracle.

`batch_probe.py` runs the actual `prepare_batch`, canonical transport serialization and loss-token-count helper on 128 variable-length samples with mixed RL/CE/reference components and routed-expert arrays. Four simulated rank partitions produce byte-identical serialized batches and identical per-rank/global component counts (4179/2230/1336). This exercises the actual CPU batch integration; it does not run a distributed collective or model.

Three-repeat CPU refinement medians for 64/8, 128/8 and 512/32 items/ranks were 0.535→0.221ms, 2.10→0.243ms and 3.050→0.123s on the local machine. These are helper timings on fixed inputs, not end-to-end training throughput. Full receipts are in `results/`; raw packed artifacts remain in the external evidence cache.

The parent independently verified every applied path, raw packed-byte equality, imported caller/harness hashes and component counts. The compact raw archive is cached with the research record (107,425 bytes; SHA256 `d3a6278deb1240f9012dd7483e42a98d281fbbd9988bb5103a054e38da120ab2`).
