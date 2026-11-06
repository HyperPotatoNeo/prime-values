# Owned trajectory unit

Implemented on the existing dependency pins before changing the Episode producer.
The critic now owns copied trajectory geometry and provenance; scoring-hook
reward timing, arrival-time evaluation, group-conditioned evaluation and
TETHER timing remain intact. Policy cleanup cannot rewrite the critic's inputs.

- Frozen `f6b9a40c`: 157 focused CPU checks passed.
- Updated unit: 182 checks passed, no failures, errors or skips.
- Eight frozen/current production comparisons matched exactly: materialized
  trajectories, evaluator requests and versions, values, independent-lambda
  targets/advantages, DPPO actor loss and full logit gradients, critic loss and
  full logit gradients, and adaptive global/positioned TETHER JSON continuation.
- The relocated GAE oracle passed 15 controls. Shared-lambda and
  context-as-action mutations were killed by assertions (8 and 15 failures);
  neither result relies on import errors or skipped cases.
- Ownership checks mutate policy tokens, masks and reward during evaluation,
  clear policy graph data before positioned TETHER, and clear policy samples
  before critic publication. Expected value targets, native action positions
  and publication versions survive.

Independent review caught and resolved two issues: preserving reward changes
from rollout-local scoring hooks, and avoiding provenance-copy overhead when
the value plane is disabled. The mathematical recurrences were moved unchanged;
the existing independent return/gradient oracles remain the intent checks.

These are CPU functional and logit-gradient comparisons with a deterministic
evaluator. They do not establish native environment, network, transformer
parameter-update or restart behavior after the forthcoming Episode migration.
No new GPU run was needed for this ownership-only unit. Value result arrays
remain mutable internally until copied and encoded for publication; replay
continues to use its established decode/read-only-packing ownership contract.

`manifest.json` records evidence hashes. `compare_credit.py` in the parent
experiment directory reproduces the eight comparisons using separate source
roots/processes and verifies production import paths. Dependency snapshots
must match each source root's Git links. Full frozen/current outputs include
source hashes and environment versions; comparisons exclude that metadata.
