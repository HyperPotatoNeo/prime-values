"""Compare saved actual producer/sink/packing outputs, with no numerical tolerances."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    receipts = {}
    for label in ["baseline", "candidate"]:
        directory = getattr(args, label)
        receipt = json.loads((directory / "result.json").read_text())
        assert receipt["status"] == "passed"
        assert receipt["raw_sha256"] == digest(directory / "raw.json")
        sources[label] = {case["mode"]: case for case in json.loads((directory / "raw.json").read_text())["cases"]}
        receipts[label] = receipt
    assert receipts["baseline"]["harness_sha256"] == receipts["candidate"]["harness_sha256"]
    assert (
        receipts["baseline"]["dependency_provenance"]["dependency_code"]
        == receipts["candidate"]["dependency_provenance"]["dependency_code"]
    )
    pairs = [
        ("baseline", "absent", "candidate", "absent"),
        ("baseline", "clipped_absent", "candidate", "clipped_absent"),
        ("candidate", "absent", "candidate", "mixed"),
        ("candidate", "absent", "candidate", "opaque"),
        ("candidate", "clipped_absent", "candidate", "clipped"),
        ("candidate", "frozen_absent", "candidate", "frozen_known"),
    ]
    comparisons = []
    for left_source, left_mode, right_source, right_mode in pairs:
        left, right = sources[left_source][left_mode], sources[right_source][right_mode]
        assert left["requests"] == right["requests"], (left_mode, right_mode, "HTTP requests")
        assert left["contexts"].keys() == right["contexts"].keys()
        for context in left["contexts"]:
            for field in ["numeric", "packed", "live_policy"]:
                assert left["contexts"][context][field] == right["contexts"][context][field], (
                    left_mode,
                    right_mode,
                    context,
                    field,
                )
        comparisons.append(
            {
                "left": f"{left_source}/{left_mode}",
                "right": f"{right_source}/{right_mode}",
                "contexts": list(left["contexts"]),
                "exact_http_requests": True,
                "exact_all_preexisting_numeric_fields": True,
            }
        )
    candidate = sources["candidate"]
    for case in candidate.values():
        for context in case["contexts"].values():
            assert all(any(sample["mask"]) for sample in context["numeric"]["samples"])
    assert candidate["mixed"]["contexts"]["task"]["sample_versions"][0].count(5) == 2
    assert 6 in candidate["mixed"]["contexts"]["task"]["sample_versions"][0]
    assert candidate["mixed"]["contexts"]["task"]["sample_versions"][1] is None
    assert candidate["clipped"]["contexts"]["task"]["sample_versions"][0][-3:] == [5, 5, 6]
    assert candidate["frozen_known"]["contexts"]["task"]["sample_versions"] == [None, None]
    result = {
        "status": "passed",
        "comparisons": comparisons,
        "exact_context_comparisons": 12,
        "baseline_result_sha256": digest(args.baseline / "result.json"),
        "candidate_result_sha256": digest(args.candidate / "result.json"),
        "harness_sha256": receipts["candidate"]["harness_sha256"],
        "comparator_sha256": digest(Path(__file__)),
        "scope": "Actual CPU HTTP fixture, native worker/TrainClient/Episode msgpack, TrainSink credit, packing and CPU tensors; no GPU/backend weight reload truth or numerical training step.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
