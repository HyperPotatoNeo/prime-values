# Native profile readiness

The current authored entrypoints are [configs/native](../../configs/native/README.md): reverse text and cached RG, each with policy-only and independent-value variants. The new group-eight smoke examples are configuration-validated; they are not a new GPU performance or convergence result.

`config-compatibility.json` explicitly declares the intended config class, any overlay base, and every environment path/ID for all 82 TOMLs under `configs/` and `examples/`. Tests independently rediscover these paths and IDs, compose tables recursively with list replacement, and check the declared parser outcome. Legacy rejection and missing native taskset packages are separate contracts. A missing eval plugin cannot hide a legacy train entry because each environment is also checked separately. No historical config is rewritten or silently mapped to a different scorer.

The 78 historical TOMLs contain 69 legacy environment entries in 44 files and 19 older native entries in 15 files (four files overlap). There are 15 distinct legacy IDs and 11 older native IDs. Those native packages are outside the selected environment extra. The four new authored examples use the two explicitly supported native tasksets. Configuration-only historical entries such as SFT, inference and SLURM overlays are not evidence that those entire runtimes/models have been exercised on this stack.

The preserved references are source/lock `1801d112` before the native-only producer transition, and `8474e835` for the earlier backend milestone. Consult their acceptance records before claiming any particular historical experiment validated. The current profile does not restore every legacy environment, tool, multimodal, Harbor or P/D workflow. That remains an explicit main-replacement decision and migration backlog; this readiness tranche does not claim main-wide compatibility.

## Validation scope

- 195 focused config/native checks pass: exhaustive catalog classification, genuine native factories, four role/sampling/value contracts, and the actual launcher TOML writer followed by child CLI reload. Existing value warmup, placement, replay, group-credit and algorithm assertions remain.
- 30 existing client/elastic tests pass after import-path-only maintenance for the target verifier's `configs.client` location; assertions are unchanged.
- A separate isolated wheel installation loads the exact pinned config/verifier/renderer dependencies and reverse-text plugin without the training wheel or root source overrides. It parses the authored policy config with a SLURM CLI override, retains `template_path=None`, and imports no heavy training modules. The slim CI job follows that exact-pin setup. The slim package minimums are conservatively tightened to these demonstrated native versions; this does not establish compatibility with every later dependency release.
- A bounded lean-CPU collection audit collected 796 selected tests and found 28 import errors: two obsolete test imports now repaired, and 26 unavailable optional training packages in that lightweight environment. These are not a full CI result. The normal CPU test selection and historical GPU matrix remain unchanged; there is no blanket skip of incompatible catalog entries.

Only the `num_workers` compatibility validator changes runtime/config behavior: it copies the caller's dictionary before consuming the deprecated alias, preserving explicit pool precedence and `auto` behavior. No value math, loss, dtype, sampling defaults, production admission or transport code changes in this tranche.
