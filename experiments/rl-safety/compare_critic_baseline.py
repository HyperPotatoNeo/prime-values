"""Require exact critic snapshots against a previously frozen, trusted fixture run."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--updated", required=True, type=Path)
    args = parser.parse_args()
    before_run = json.loads((args.baseline / "result.json").read_text())
    after_run = json.loads((args.updated / "result.json").read_text())
    assert before_run == after_run, "Different rank/layout fixture"
    compared = {}
    for name in ("initial", "step1", "step2"):
        before = torch.load(args.baseline / f"{name}.pt", weights_only=False)
        after = torch.load(args.updated / f"{name}.pt", weights_only=False)
        assert before.keys() == after.keys()
        for key in ("version", "total_tokens", "total_samples", "lr", "scheduler"):
            assert before[key] == after[key], (name, key)
        for group in ("model", "moments"):
            assert before[group].keys() == after[group].keys()
            for key in before[group]:
                assert torch.isfinite(after[group][key]).all(), (name, group, key)
                assert torch.equal(before[group][key], after[group][key]), (name, group, key)
        compared[name] = {group: len(after[group]) for group in ("model", "moments")}
    print(json.dumps({"world_size": after_run["world_size"], "exact_matches": compared}, indent=2))


if __name__ == "__main__":
    main()
