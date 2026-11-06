# Parameter-safe activation offloading

This unit adapts upstream `53aecbf10ed0be79555c26a7a66f5c56218f0fa8`. Autograd can save a plain tensor view of a model parameter, including a local view of a distributed expert parameter. Checking only whether the saved object is a `Parameter` does not recognize that storage. Such views must stay on the model device while genuine activations remain eligible for CPU offloading.

`OffloadActivations` receives the model and records its parameter slots as `(module, name)`. At each eligible save it reads the current parameter in each slot, obtains a DTensor's local tensor when needed, and excludes matching storage. It does not cache parameter objects or storage identities across FSDP unsharding and resharding. Policy RL, SFT and the independent value trainer use the same helper.

Activation size thresholds, pinned-memory configuration, stream-disabled behavior, checkpointing settings, optimization/reduction dtypes and loss computations are unchanged. This does not add activation-copy deduplication: repeated saves of mutable shared storage require separate lifecycle analysis. No inference path is changed.

Acceptance requires a negative control for parameter-view offloading and a positive control showing that an activation clone is still offloaded. Live parameter replacement and actual DTensor/FSDP execution must exercise the storage lifetime. A compiled critic comparison must retain full-gradient and update behavior. Numerical and copy-volume results belong in their individual receipts; the source change alone does not establish an end-to-end throughput or memory improvement.

## Completed acceptance

All eight stages exited normally. A real initialized GroupedExperts block used expert parallelism across two ranks, nested FSDP, BF16 forward, FP32 master parameters/reductions/Adam, and six captured compiled graphs per rank. Both the frozen and corrected offloader matched the offload-disabled control exactly for all 48 saved tensors per rank, including complete gradients and optimizer states over two updates.

The frozen offloader copied six expert parameter aliases (196,608 bytes per rank); the corrected helper copied none. Both still copied 16 real activations (126,976 bytes per rank). Live FSDP storage replacement was observed; an explicit replaced-parameter control and an equal-valued activation clone distinguish ownership from tensor contents. These are saved-event byte counts, not peak-memory or throughput measurements.

The actual compiled two-rank ValueTrainerRuntime used identical full activation checkpointing in all three controls. Frozen offload-off, frozen offload-on and corrected offload-on matched exactly: 22 initial tensors and 110 tensors after each of two updates, including full pre-step gradients, FP32 model state and Adam moments. The run included the real masked dummy rank and nonzero updates. Both ranks executed two compiled blocks/graphs and offloaded real activations. No value objectives, normalization or precision settings changed.

Prelaunch review caught that enabling offload implicitly enables activation checkpointing. The harness explicitly pins the same checkpointing configuration across controls and checks the resolved configs; no failed GPU run or relaxed comparison was needed. The upstream copy-deduplication change is intentionally excluded because alias mutation requires separate snapshot semantics.

The parent independently verified 96 archive members, 332 matching runtime source/config/test files, the seven applied patch paths and all saved paired tensors. One additional source file differs only in the separately committed Rollout docstring; its frozen hash and executable AST identity were checked. The original source-review receipts retain their pre-GPU status, with later acceptance recorded separately.

Raw evidence: `prime-values-shared:/data/activation-offload-001/evidence.tar.gz`, 51,039,501 bytes, SHA256 `0aca221e61c5c1ad2f1fca1cfa9a8e187917841e4c7d4e9bf83e9bef65ffdc83`. Paired tiny-model states and raw logs are included; no large-model pipeline or full MoE throughput result is claimed. The existing eight-GPU job was reused; stage duration does not establish utilization.
