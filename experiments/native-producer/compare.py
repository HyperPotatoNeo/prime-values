"""Compare independently collected frozen/native production cohorts exactly."""

import argparse
import hashlib
import json
from pathlib import Path


def differences(a, b, path=""):
    if type(a) is not type(b):
        yield path, repr(a)[:180], repr(b)[:180]
    elif isinstance(a, dict):
        for k in sorted(a.keys() | b.keys()):
            if k not in a or k not in b:
                yield (
                    path + "/" + k,
                    "missing" if k not in a else repr(a[k])[:180],
                    "missing" if k not in b else repr(b[k])[:180],
                )
            else:
                yield from differences(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        if len(a) != len(b):
            yield path + "/length", len(a), len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            yield from differences(x, y, path + "/" + str(i))
    elif a != b:
        yield path, repr(a)[:180], repr(b)[:180]


def compare(baseline, candidate, output):
    b = json.loads((baseline / "raw.json").read_text())
    c = json.loads((candidate / "raw.json").read_text())
    br = json.loads((baseline / "result.json").read_text())
    cr = json.loads((candidate / "result.json").read_text())
    assert br["status"] == cr["status"] == "passed"
    assert br["provenance"]["harness_sha256"] == cr["provenance"]["harness_sha256"]
    assert hashlib.sha256((baseline / "raw.json").read_bytes()).hexdigest() == br["raw_sha256"]
    assert hashlib.sha256((candidate / "raw.json").read_bytes()).hexdigest() == cr["raw_sha256"]
    bc = {(x["env"], x["case"]): x for x in b["cases"]}
    cc = {(x["env"], x["case"]): x for x in c["cases"]}
    assert bc.keys() == cc.keys()
    failures = []
    passed = []
    for key in sorted(bc):
        payload_fields = ["idx", "prompt", "system_prompt", "answer"] + (
            ["variant_id", "scorer_entry"] if key[0] == "rg" else []
        )
        failures.extend(
            differences(
                {k: bc[key]["payload"][k] for k in payload_fields},
                {k: cc[key]["payload"][k] for k in payload_fields},
                "/".join(key) + "/selected_payload",
            )
        )
        for field in ["records", "credit"]:
            diff = list(differences(bc[key][field], cc[key][field], "/".join(key) + "/" + field))
            failures.extend(diff)
            if not diff:
                passed.append("/".join(key) + "/" + field)
    expected_requests = {json.dumps(x, sort_keys=True) for x in b["requests"]}
    actual_requests = {json.dumps(x, sort_keys=True) for x in c["requests"]}
    if expected_requests != actual_requests:
        failures.extend(("requests/baseline-only", x, "missing") for x in expected_requests - actual_requests)
        failures.extend(("requests/candidate-only", "missing", x) for x in actual_requests - expected_requests)
    else:
        passed.append("literal policy request body sets (counts differ for target controls)")
    report = {
        "status": "passed" if not failures else "failed",
        "exact_comparisons": passed,
        "difference_count": len(failures),
        "differences": failures,
        "baseline_raw_sha256": br["raw_sha256"],
        "candidate_raw_sha256": cr["raw_sha256"],
        "harness_sha256": br["provenance"]["harness_sha256"],
        "scope": "Actual value-owned geometry, sampled logprobs, task/peer credit and TETHER continuation; literal requests. Deterministic HTTP/evaluator fixtures, no model/GPU numerical inference claim.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "differences"}, indent=2))
    assert not failures, f"{len(failures)} differences; see {output}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    compare(a.baseline, a.candidate, a.output)
