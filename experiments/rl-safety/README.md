# RL safety and reliability validation

This milestone combines numerical safety and operational reliability changes while keeping the existing policy objective and independent value plane. The pipeline check runs actual Qwen generation, policy training, critic training and serving, then restarts all components from saved policy and critic checkpoints.

## Numerical and failure checks

Run the following serially before starting any live pipeline in the test pod:

```bash
bash experiments/rl-safety/run_checks.sh /data/safety-checks
uv run --no-sync python experiments/rl-safety/distributed_checks.py --output /data/safety-distributed
```

The first command combines the existing value baseline, focused reliability and
streaming-summary tests, and seven assertion-only mutation checks. The second
runs the fatal-rank/healthy-shutdown contrast plus the one/two-rank analytical
policy-normalization and diagnostic-summary probe. `--baseline-source PATH`
optionally runs the fatal probe against an extracted baseline `src/` tree; it
requires the old implementation to reach the intentional failure and time out,
while the fixed implementation must report that failure and exit promptly.

Repeat the real critic commands from
[the value baseline](../value-baseline/README.md), then compare the resulting
snapshots directly with the frozen baseline at each rank count:

```bash
uv run --no-sync python experiments/rl-safety/compare_critic_baseline.py \
  --baseline /data/frozen/rank-1 --updated /data/current/rank-1
uv run --no-sync python experiments/rl-safety/compare_critic_baseline.py \
  --baseline /data/frozen/rank-2 --updated /data/current/rank-2
uv run --no-sync python experiments/rl-safety/capture_source.py /data/source.json
```

The direct comparison requires exact model, optimizer, scheduler and progress
matches for the initial and two updated states. Only load trusted local fixture
artifacts. Source capture verifies actual imported files and records dependency
versions and source hashes; compare those hashes with the reviewed commit.

## Real pipeline check

Run in a fresh isolated pod with at least four visible GPUs, the frozen repository environment and its submodules. The built-in native `taskset.id = "reverse-text-v1"` loads `PrimeIntellect/Reverse-Text-RL` with the tool-free `null` harness and subprocess runtime; the model defaults to `Qwen/Qwen3-4B-Instruct-2507`. Dataset/model caches may be shared, but source, output and test processes must be separate from active training.

```bash
uv run --no-sync python experiments/rl-safety/pipeline.py --output /data/safety-dedicated --placement dedicated
uv run --no-sync python experiments/rl-safety/pipeline.py --output /data/safety-colocated --placement trainer
```

For faster checkpoint I/O, add `--checkpoint-root /tmp/safety-dedicated-checkpoints` with a fresh node-local path. Only `run/checkpoints`, `run/weights`, and `run/value/checkpoints` become symlinks; requested/resolved configs, logs, scalar snapshots and evidence remain under the persistent `--output` path. The harness refuses existing checkpoint storage and records the routing in `checkpoint-storage.json`. Both fresh and resumed stages must run in the same pod with that storage intact. These large checkpoint artifacts are temporary and can be discarded after validation; removing the pod loses them and leaves the persistent symlinks dangling. The retained manifests and scalar snapshots are evidence, not resumable model checkpoints.

If the fresh stage finished but a harness validation error prevented restart, run the corrected harness with `--output EXISTING_OUTPUT --continue-from-fresh`. It rereads the original requested/resolved configs and archived actual child configs, then reruns every fresh-stage check against archived logs, live-version observations and existing checkpoints before launching the resumed stage. It requires no existing resumed directory and derives model/placement from the saved configuration. Do not pass preparation or checkpoint-storage options; existing local checkpoint storage must remain available. The orchestrator checkpoint is located from its resolved output directory (normally `run/run_default`), separately from actor checkpoints, and receives its own small manifest.

Every new stage records its natural launcher exit before any harness cleanup and must exit zero. Historical fresh-stage recovery without that receipt is explicitly qualified in the result; completion logs and checkpoints do not retroactively prove its exit code. The resumed stage always requires recorded exit zero. The only tolerated traceback is the observed vLLM `EngineDeadError` during shutdown: it must be in `inference.log`, after launcher completion and the API shutdown marker, with application shutdown completion after every terminal exception. Timestamp ordering applies to these bounded, same-clock stage logs. Other exceptions, earlier traces and incomplete final shutdowns fail validation. `check_log_controls.py` exercises this distinction against actual logs and adversarial edits and saves hashed-input results.

Before a stage starts, any existing live `run/logs` directory is moved into that stage's `prelaunch-logs` archive. The launcher creates a new live log directory, so startup monitoring cannot misinterpret the previous stage's shutdown traces as current failures. Prior logs remain preserved in both their original stage archive and this transition archive.

Use different fresh output directories for attempts and placements. `--prepare-only` validates the top-level configuration and writes actual launcher TOMLs, then reloads every component through its CLI parser and checks equality; its output path must also be fresh. It also constructs the actual native environment server on an ephemeral local port, loads the dataset and checks its task contract, then closes the server before any GPU process starts. The local `rl` entrypoint manages the policy trainer, inference, critic trainer and dedicated evaluator (or trainer-colocated evaluator). The script creates the vLLM IPC directory before launch, scans nested environment and component logs for tracebacks, uses process-group termination on failures/timeouts, and retains separate fresh/resumed logs and actual child configs. Evaluator version probes decode the service's MessagePack response.

The experiment uses 512-token sequences, 64-token generation limits, batch eight, group two and two critic warmup updates. It runs three policy updates, then launches a fresh process to resume at policy step three and the actual latest stable independent critic version, continuing to policy step four. Model compilation, activation checkpointing and activation offloading retain existing defaults; optimizer offload is disabled. Optimization/reduction precision is unchanged. The local launcher omits optional `None` sections when writing TOML, so a default-enabled section cannot currently be disabled by passing JSON null through this launcher. The component roundtrip guard detects that mismatch before training.

Evidence includes requested and resolved configurations, live evaluator-version observations, per-component logs, finite logged policy/critic losses and actor gradient norms, dedicated evaluator weight adoption, policy DCP metadata/shards, and stable exported-weight/critic checkpoints. The final summary requires actual policy updates, actual critic updates, a warmup wait, an evaluator reaching version two, and policy/orchestrator/critic resume messages. The checkpoint/version checks establish basic lifecycle progress, not exact numerical equivalence between independent stochastic runs or evaluator placements. Exact shipped-cohort warmup provenance remains covered by focused unit tests; observing a live endpoint alone is not that proof.

Policy and critic checkpoint retention is one; policy exported weights inherit the same retention. Each stage records a checkpoint file/size inventory with SHA-256 hashes of metadata, progress and marker files before the next stage can retire its checkpoints. Full optimizer shards are retained only for the latest checkpoint and are not redundantly hashed or copied into evidence.

While training, a CPU-only DCP reader requests just the three actor progress scalars and saves them separately. Verification requires an intermediate step-one checkpoint with completed-update counters, checks cumulative samples against the actual logged packed microbatch counts, checks growing token counts, and checks restored counters across restart. These counters describe packed microbatches and padded tokens, not raw rollout counts. Checking only the final checkpoint would miss the historical in-loop counter lag.

Do not run pytest concurrently: the repository test cleanup fixture terminates matching training processes. Run the numerical/mutation suite first or after all pipeline components exit. The pipeline's 30-minute per-stage deadline is a bound, not an expected duration.

## Limits

This is a lifecycle/correctness smoke run, not a reward or throughput benchmark. It does not test all branch/privileged-context modes, adverse network conditions, multi-node execution, or every model family. Preserve the fixed numerical tests alongside it, inspect real trajectories and metrics, and evaluate any failure before expanding the migration. Passing exit codes and checkpoint existence alone do not establish useful learning.
