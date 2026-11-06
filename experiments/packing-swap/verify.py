"""Compare actual pure packing helpers against a frozen source, without GPU imports."""

import argparse
import ast
import copy
import hashlib
import heapq
import json
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def load_helpers(path):
    names = {
        "_WeightedSet",
        "_KKState",
        "_karmarkar_karp",
        "_partition_loads",
        "_refine_by_swapping_scalar",
        "_refine_by_swapping",
        "balanced_partition",
    }
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    scope = dict(globals(), Sequence=Sequence, dataclass=dataclass, field=field, heapq=heapq, np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    before, after = load_helpers(args.baseline), load_helpers(args.candidate)
    fallback_calls = []
    scalar = after["_refine_by_swapping_scalar"]

    def observe_fallback(weights, partitions):
        fallback_calls.append(True)
        return scalar(weights, partitions)

    after["_refine_by_swapping_scalar"] = observe_fallback
    cases = [
        ("improving-tie", [10, 10, 0, 0, 0, 0], [[0, 1], [2, 3], [4, 5]], False),
        ("ties", [1] * 12, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], False),
        ("empty-partitions", [8, 3, 7], [[0], [], [1, 2]], False),
        ("empty-items", [], [[], []], False),
        ("one-partition", [3, -7, 2], [[0, 1, 2]], False),
        ("huge-weights", [10**100, 1, 10**100 - 1, 2], [[0, 1], [2, 3]], True),
        ("unsafe-spread", [2 * 10**18] * 6 + [1, 2], [list(range(6)), [6, 7]], True),
        ("large-totals-safe-relative", [10**18 + i for i in range(24)], [list(range(12)), list(range(12, 24))], False),
        ("negative-weights", [-8, 3, -7, 2, 6, -1], [[0, 1], [2, 3], [4, 5]], False),
        ("non-integer-fallback", [1.5, 8.5, 2.5, 6.5], [[0, 1], [2, 3]], True),
    ]
    rng = random.Random(911)
    for index in range(120):
        ranks = rng.randint(2, 7)
        count = ranks * rng.randint(1, 5)
        weights = [rng.randint(-10 if index % 3 == 0 else 0, 100) for _ in range(count)]
        indices = list(range(count))
        rng.shuffle(indices)
        cases.append((f"seeded-{index}", weights, [indices[r::ranks] for r in range(ranks)], False))
    outcomes = []
    for name, weights, partitions, expect_fallback in cases:
        original = copy.deepcopy((weights, partitions))
        fallback_calls.clear()
        expected = before["_refine_by_swapping"](weights, partitions)
        actual = after["_refine_by_swapping"](weights, partitions)
        assert actual == expected, (name, actual, expected)
        if name == "improving-tie":
            assert actual == [[2, 1], [0, 3], [4, 5]]
        assert (weights, partitions) == original, name
        assert bool(fallback_calls) == expect_fallback, name
        outcomes.append({"case": name, "exact_ordered_partitions": True, "scalar_fallback": bool(fallback_calls)})
    public_cases = 0
    for ranks in (1, 2, 3, 8):
        for multiplier in (1, 2, 5):
            weights = [rng.randrange(10**12) for _ in range(ranks * multiplier)]
            original = weights.copy()
            assert after["balanced_partition"](weights, ranks) == before["balanced_partition"](weights, ranks)
            assert weights == original
            public_cases += 1
    timings = []
    for count, ranks in ((64, 8), (128, 8), (512, 32)):
        weights = [rng.randrange(1, 10**12) for _ in range(count)]
        partitions = before["_karmarkar_karp"](weights, ranks)
        expected = before["_refine_by_swapping"](weights, partitions)
        row = {"items": count, "ranks": ranks, "repeats": 3}
        for label, scope in (("frozen", before), ("candidate", after)):
            samples = []
            for _ in range(3):
                started = time.perf_counter()
                actual = scope["_refine_by_swapping"](weights, partitions)
                samples.append(time.perf_counter() - started)
                assert actual == expected
            row[f"{label}_seconds"] = samples
            row[f"{label}_median_seconds"] = statistics.median(samples)
        timings.append(row)
    receipt = {
        "source_files": {
            label: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for label, path in (("baseline", args.baseline), ("candidate", args.candidate))
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "direct_cases": outcomes,
        "public_balanced_partition_cases": public_cases,
        "timings": timings,
        "scope": "Actual pure helper ASTs extracted from frozen/current sources; exact ordered partitions and nonmutation, arbitrary-integer fallback controls. CPU helper timings only; no model, GPU or end-to-end throughput claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"direct_cases": len(outcomes), "public_cases": public_cases, "timings": timings}))


if __name__ == "__main__":
    main()
