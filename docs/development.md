# Development

This page covers workflows for developing on `prime-rl` itself — running the test suite, contributing changes, and adding new model architectures with the small-scale tooling we use to iterate on MoE families without booting up a 100B+ run.

## Table of Contents

- [Test Suite](#test-suite)
  - [Layout](#layout)
  - [Running Tests Locally](#running-tests-locally)
  - [CI Workflows](#ci-workflows)
  - [Markers](#markers)
- [Pre-Commit Hooks](#pre-commit-hooks)
- [Adding a New Model](#adding-a-new-model)
  - [Implement the Modeling Code](#implement-the-modeling-code)
  - [Register a Mini Preset](#register-a-mini-preset)
  - [Run the Smoke Test](#run-the-smoke-test)

## Test Suite

The test suite is split into three tiers, each with its own CI workflow.

### Layout

- **`tests/unit/`** — fast-running, hermetic tests for isolated logic: config parsing and validation, advantage / loss / scheduler / packer math, individual dataset paths, model-conversion roundtrips, etc. Tests that need a GPU are tagged with the `gpu` marker.
- **`tests/integration/`** — full-stack RL/SFT runs on a tiny model end-to-end through inference + orchestrator + trainer.
- **`tests/nightly/`** — preserved scenario assertions available through the explicitly pinned manual workflow; these do not run nightly on the native profile.

### Running Tests Locally

```bash
uv run pytest tests/unit -v -m "not gpu"                    # CPU unit suite
uv run pytest tests/unit -v -m gpu                          # GPU unit suite
uv run pytest tests/integration/test_native_profile.py -vvs # native policy/value lifecycle
```

Use the manual, pinned workflow for preserved scenario tests; a broad integration-directory sweep is not the current native acceptance command.

### CI Workflows

| Workflow | Trigger | What runs | Where |
|---|---|---|---|
| [`cpu_tests.yaml`](../.github/workflows/cpu_tests.yaml) | every PR + push to `main` | All CPU unit tests, plus an isolated exact-pin config/plugin wheel install and native CLI check without heavy trainer/inference imports | `ubuntu-latest` |
| [`gpu_tests.yaml`](../.github/workflows/gpu_tests.yaml) | repository opt-in for non-draft PRs/pushes, or manual dispatch | GPU unit tests and native reverse-text policy/value fresh-resume integration | self-hosted GPU runners (`vm`, `4xa6000`) |
| [`nightly_tests.yaml`](../.github/workflows/nightly_tests.yaml) | manual only | Explicitly selected preserved test on fixed source/lock `1801d112`; no current-branch compatibility claim | self-hosted GPU runners |

The GPU workflow requires configured self-hosted runners labeled `vm` and `4xa6000`. Enable automatic execution with repository variable `ENABLE_GPU_CI=true`, or launch it manually with `workflow_dispatch`; automatic PR runs still skip drafts. Without runners, run `uv run pytest tests/integration/test_native_profile.py -vvs` on an appropriately provisioned local machine. A skipped job is not acceptance. External H200 validation is separate evidence and does not establish A6000 memory fit. RG's external artifact is not provisioned by Actions. The recorded [acceptance index](../experiments/README.md) distinguishes completed runtime evidence from pending workflow execution.

### Markers

Two pytest markers are declared in `pyproject.toml` (`addopts = "--strict-markers"`):

- `gpu` — gate a test that needs CUDA. CPU CI uses `-m "not gpu"`; the GPU unit job uses `-m gpu`.
- `slow` — gate a test that's expensive enough you'd usually skip it locally. Deselect with `-m "not slow"`.

## Pre-Commit Hooks

Install the [pre-commit](https://pre-commit.com) hooks before your first commit so ruff check + format run on staged Python files automatically:

```bash
uv run pre-commit install
```

## Adding a New Model

Bringing up a new model family is three steps: implement the modeling code, register a mini preset, and run the smoke test. The preset and smoke test let you iterate on the modeling code at ~0.5B scale on 1–2 GPUs instead of paying the cost of the full-size model — useful for catching bugs in modeling code, state-dict conversions, and pipeline integration before scaling.

### Implement the Modeling Code

Drop the modeling code under `src/prime_rl/trainer/models/<arch>/` (HF-compatible config, modeling, and weight conversion). Mirror the layout of an existing family — `glm4_moe/` or `qwen3_moe/` are good starting points.

### Register a Mini Preset

Add an entry to [`scripts/mini_moe.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/scripts/mini_moe.py) so the smoke-test workflow can build a ~0.5B test model in your architecture. The preset names the config class, picks small dimensions, and wires up the HF + PrimeRL model classes plus a tokenizer source:

```python
ARCH_PRESETS = {
    "glm4_moe": {
        "config_class": Glm4MoeConfig,
        "config_kwargs": dict(hidden_size=1024, num_hidden_layers=24, n_routed_experts=8, ...),
        "hf_model_class": HFGlm4MoeForCausalLM,
        "prime_model_class": PrimeRLGlm4MoeForCausalLM,
        "tokenizer_source": "THUDM/GLM-4-9B-0414",
    },
    # add your arch here
}
```

### Run the Smoke Test

Build the mini model. This creates a ~543M-parameter GLM-4 MoE (1024 hidden, 24 layers, 8 experts) with random weights, copies the tokenizer from the original GLM-4 model, and verifies the HF↔PrimeRL roundtrip is lossless:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe
```

To re-verify the roundtrip after a modeling-code change without re-creating the model:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe --verify-only
```

Warm up the random-weight mini model with SFT on reverse-text so KL divergence becomes meaningful in the RL phase. Loss drops from ~12 to ~2.5 — the output won't be coherent, but the distribution is non-trivial. A pre-built SFT'd checkpoint lives at [samsja/mini-glm-moe](https://huggingface.co/samsja/mini-glm-moe) if you want to skip this step:

```bash
uv run sft @ configs/debug/moe/sft/train.toml \
  --model.name ./mini-glm-moe \
  --data.name PrimeIntellect/Reverse-Text-SFT \
  --data.type null \
  --max_steps 200 \
  --optim.lr 1e-4 \
  --ckpt.weights
```

Then run the full RL stack on reverse-text:

```bash
uv run rl @ configs/ci/integration/reverse_text_moe/start.toml \
  --model.name samsja/mini-glm-moe \
  --trainer.model.impl custom \
  --inference.gpu-memory-utilization 0.7 \
  --inference.model.max-model-len 2048
```

What to look for:

- **No crashes.** Validates the full inference + orchestrator + trainer pipeline end-to-end.
- **Finite, non-zero KL.** Confirms the reference distribution is meaningful.
- **Loss reasonable.** Not NaN, not stuck.

Don't expect reward to climb meaningfully in 20 steps on a random model.

### Requirements for merging a new model

Before merging a new model, you need to ensure the following:

- The model is correctly registered and defines and all the required methods - such as `convert_hf_layer_to_tt` and `convert_tt_layer_to_hf`.
- The small smoke test passes.

In the PR that adds the new model, you also need to provide a table covering the KL mismatch across 20 steps on `math` environment with `batch_size=64`. All the entries in the table must lower than 0.015. If this is not met, the PR will not be merged (unless reasonable justification is provided). This is to ensure all our models are consistent and their implementations match the implementations in the inference framework.
