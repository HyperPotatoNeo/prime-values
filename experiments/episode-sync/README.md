# Episode synchronization

This work continues the verified safety/reliability baseline `f6b9a40c` on
`feat/siddarth-episode-sync`. The full upstream reference is `3c96816d`;
the first producer milestone is `cb74ae2b`, with Verifiers `192fd8ef` and
Renderers `bdb96b0c`. That boundary introduces real Episodes while retaining
upstream's legacy environment bridge and the current GPU dependency family.
It does not claim compatibility with all of latest upstream.

## Units

1. Own critic inputs independently of mutable policy samples. Capture tokens,
   unique training masks, native action provenance, canonical branch layout,
   reward, prefix, and policy version before scoring/filtering. Extract the
   existing credit calculation without changing its arithmetic. Critic
   publication and positioned TETHER consume that same snapshot.
2. Connect the pinned canonical Episode producer through one explicit
   single-trace projection. Check the original episode before selecting
   trainable traces; reject unsupported multi-trace credit semantics. Preserve
   episode failure/retry provenance, native task data, and the maintained
   legacy bridge. Validate actual generation, independent critic publication,
   policy updates and restart with both evaluator placements.
3. Advance task ownership, payload transport/packing and shared trainer
   services in coordinated units. Preserve critic clocks and value-head
   publication, checkpoint semantics and supported environments. Upgrade the
   GPU software bundle coherently when required by those services.

New objectives, ratio caps, weighted normalization, staleness admission and
zero-advantage pruning are separate decisions. This plumbing work preserves
optimization/reduction precision, ancestral sampling, DPPO/KL defaults,
independent GAE/target lambdas, and TETHER's apply-before-observe timing.

## Evidence and review

Use the existing independent numerical oracles for mathematical intent, plus
frozen-versus-current production comparisons for accidental behavior changes.
Target ownership checks at policy mutation/release, privileged prefix
projection, shared action prefixes, mixed critic versions and group barriers.
Actual model/run checks cover the boundaries a CPU fixture cannot establish.
Do not interpret passing finite examples as proof of every environment or
algorithm, nor successful training as measured reward improvement.

The first local check used isolated Python 3.12, Torch 2.11 and Transformers
5.6.2, with the baseline's exact verifier/renderer pins. Its other CPU
dependencies were independently resolved; it complements the previously
recorded frozen cluster evidence rather than reproducing that environment.
The unmodified baseline passed 157 focused checks. Results for each completed
unit are recorded beside this document and in the private Resolution research
cache under `projects/prime-values/`.
