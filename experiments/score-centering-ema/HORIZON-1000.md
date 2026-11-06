# Comparison horizon: 1,000 updates

On November 6, Siddarth shortened the ten-experiment comparison from 2,000 to
1,000 trainer updates. Keep the original histories, including steps above 1,000;
use the common 1–1,000 prefix for consolidated comparisons. No loss, sampling,
batch, sequence-length, optimizer, precision or parallelism settings changed.
The constant learning-rate schedule is independent of the horizon.

The old local `queue-224-fast1` supervisor was terminated before stopping Jobs.
Do not restart it: its launch configs still specify 2,000 updates. Six runs were
already beyond the new horizon and were intentionally stopped:

| Experiment | Seed | Last observed completed trainer step |
| --- | --- | ---: |
| Control | 11 | 1220 |
| Control | 22 | 1369 |
| Score centering, top 128 | 22 | 1078 |
| EMA beta 0.003 | 11 | 1474 |
| EMA beta 0.01 | 11 | 1526 |
| EMA beta 0.03 | 11 | 1482 |

Score-centering seed 11 reached exactly 1,000 completed updates, saved its trainer
checkpoint, and was intentionally stopped at 19:55 UTC on November 6.
`USER_STOPPED.json` in each output directory records intentional cancellation;
signal-handler `failure.txt` files do not turn this into a training failure.
No natural-completion flags are fabricated.

The three remaining EMA seed-22 runs (beta 0.003, 0.01 and 0.03) were freshly
submitted from `prepared-1000/`. Both trainer and orchestrator limits are 1,000.
Each uses two trainer and two inference nodes (32 H200s); together they reserve
96 GPUs. All retain online W&B logging under `swish/siddarth-prime-values`.
The ten-run identity and replacement-attempt map remains in
`prepared-fast1/manifest.json`; `prepared-1000/manifest.json` is a three-run
launch overlay, not a new experiment matrix.

Validation: all twelve new JSON files match the original configs except the
explicit 2,000 → 1,000 max-step fields; nine actual child CLI parses passed;
all twelve local/remote JSON hashes matched; all fifteen Kubernetes objects
passed server dry run. Services were applied separately before Job creation.
Training code remains the validated runtime through `2df6fa1a`.

Full logs and retained checkpoints remain under `/data/outputs/<run-name>` on
`prime-values-shared`. Checkpoint retention was two: some runs that passed 1,200
have already pruned their exact step-1,000 weights. Their scalar histories are
preserved, so the common-horizon learning-curve comparison remains available;
do not claim every arm has exact step-1,000 weights for a new held-out evaluation.

Operational receipts are in local `results/horizon-1000/` and the corresponding
private research-cache record. Preserve raw scheduler dumps locally; publish
only allowlisted resource receipts without environment values or credentials.
