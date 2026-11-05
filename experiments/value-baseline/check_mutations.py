"""Challenge selected numerical oracles in a disposable source copy.

Run serially in a dedicated test pod: the repository's pytest cleanup fixture
terminates torchrun processes. An import/collection failure is never a kill.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

GAE = "tests/unit/orchestrator/test_advantage.py::test_gae_matches_finite_horizon_return_mixture_with_context_gaps"
POLICY = "tests/unit/train/rl/test_loss.py::test_policy_loss_matches_tabulated_probability_objective_and_logit_gradient"
MIXED = "tests/unit/train/rl/test_loss.py::test_mixed_component_gradient_is_invariant_to_sequence_partition"
VALUE = "tests/unit/value/test_model_parity.py::test_critic_packing_serving_and_gradients_match_independent_sequences"

# name, source, exact old text, exact replacement, expected occurrences, tests
MUTATIONS = [
    (
        "shared_lambdas",
        "prime_rl/orchestrator/algo/advantage.py",
        "gamma * value_target_lambda * nonterminal * next_target_advantage",
        "gamma * gae_lambda * nonterminal * next_target_advantage",
        1,
        [GAE],
    ),
    (
        "context_as_action",
        "prime_rl/orchestrator/algo/advantage.py",
        "[index for index, trainable in enumerate(mask) if trainable]",
        "list(range(len(mask)))",
        1,
        [GAE],
    ),
    (
        "symmetric_dppo",
        "prime_rl/trainer/rl/loss.py",
        "torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)",
        "(dppo_invalid_mask_high | dppo_invalid_mask_low)",
        1,
        [POLICY],
    ),
    (
        "drop_rejected_action_kl",
        "prime_rl/trainer/rl/loss.py",
        "kl_loss = loss_mask * log_importance_ratio**2",
        "kl_loss = keep_mask * log_importance_ratio**2",
        3,
        [POLICY, MIXED],
    ),
    (
        "differentiate_reference_advantage",
        "prime_rl/trainer/rl/loss.py",
        "ref_kl.detach() * importance_ratio",
        "ref_kl * importance_ratio",
        1,
        [MIXED],
    ),
    (
        "gate_ce_by_action_mask",
        "prime_rl/trainer/rl/loss.py",
        "ce_mask = ce_w != 0",
        "ce_mask = mask & (ce_w != 0)",
        1,
        [MIXED],
    ),
    (
        "unshifted_value_state",
        "prime_rl/value/math.py",
        "aligned = flat.roll(1, dims=0).index_copy(0, starts, boundary_values)",
        "aligned = flat.index_copy(0, starts, boundary_values)",
        1,
        [VALUE],
    ),
]


def junit_result(path):
    if not path.exists():
        return {"report_error": "missing JUnit report"}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        return {"report_error": str(error)}
    cases = list(root.iter("testcase"))
    failures = [failure for case in cases for failure in case.findall("failure")]
    return {
        "tests": len(cases),
        "errors": sum(len(case.findall("error")) for case in cases),
        "skips": sum(len(case.findall("skipped")) for case in cases),
        "failures": len(failures),
        "assertion_failures": sum(
            failure.get("type", "").endswith("AssertionError")
            or failure.get("message", "").startswith(("assert ", "AssertionError"))
            or bool(re.search(r"^E\s+(?:AssertionError\b|assert\s)", failure.text or "", re.MULTILINE))
            for failure in failures
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    results = []
    with tempfile.TemporaryDirectory(prefix="value-baseline-mutations-") as scratch:
        scratch = Path(scratch).resolve()
        source = scratch / "src"
        shutil.copytree(repo / "src", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        expected = {path.removesuffix(".py").replace("/", "."): str(source / path) for _, path, *_ in MUTATIONS}
        # Run the guard inside each pytest process, before collection, so an
        # editable-install path cannot silently bypass the mutation.
        (scratch / "mutation_source_guard.py").write_text(
            "import importlib, json, os\nfrom pathlib import Path\n"
            "def pytest_sessionstart(session):\n"
            f"    expected = {expected!r}\n"
            "    actual = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in expected}\n"
            "    Path(os.environ['MUTATION_SOURCE_REPORT']).write_text(json.dumps(actual, indent=2))\n"
            "    if actual != expected:\n"
            "        raise RuntimeError(f'Mutation source precedence failed: {actual!r} != {expected!r}')\n"
        )
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        paths = [str(scratch), str(source), str(repo / "packages/prime-rl-configs/src"), str(repo)]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)

        def run(name, tests, control=False):
            destination = output / name
            destination.mkdir(exist_ok=True)
            for report in ("pytest.xml", "imported_sources.json"):
                (destination / report).unlink(missing_ok=True)
            env["MUTATION_SOURCE_REPORT"] = str(destination / "imported_sources.json")
            command = [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-B",
                "-m",
                "pytest",
                "-q",
                "-p",
                "mutation_source_guard",
                *tests,
                f"--junitxml={destination / 'pytest.xml'}",
            ]
            with (destination / "pytest.log").open("w") as log:
                try:
                    process = subprocess.Popen(
                        command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    code = process.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = "timeout"
            result = {"name": name, "exit_code": code, **junit_result(destination / "pytest.xml")}
            verified = (destination / "imported_sources.json").exists()
            if verified:
                verified = json.loads((destination / "imported_sources.json").read_text()) == expected
            valid = verified and result.get("tests", 0) > 0 and result.get("errors") == result.get("skips") == 0
            if control:
                result["status"] = "passed" if valid and code == 0 and result["failures"] == 0 else "control_failed"
            elif valid and code == 1 and result["failures"] > 0 and result["failures"] == result["assertion_failures"]:
                result["status"] = "killed"
            elif valid and code == 0 and result["failures"] == 0:
                result["status"] = "survived"
            else:
                result["status"] = "infrastructure_or_nonassertion_failure"
            results.append(result)
            (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            return result["status"]

        tests = list(dict.fromkeys(test for *_, tests in MUTATIONS for test in tests))
        if run("control", tests, control=True) != "passed":
            return 1
        for name, path, old, new, count, tests in MUTATIONS:
            target = source / path
            original = target.read_text()
            if original.count(old) != count:
                results.append({"name": name, "status": "mutation_not_applicable", "expected_occurrences": count})
                (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
                continue
            target.write_text(original.replace(old, new))
            try:
                run(name, tests)
            finally:
                target.write_text(original)
    return int(any(result["status"] not in ("passed", "killed") for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
