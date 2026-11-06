"""Two real Gloo ranks: unequal populations, empty rank and CP observation ownership."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import torch.distributed as dist

from prime_rl.trainer.rl.metrics import accumulate_policy_provenance, compute_policy_provenance
from prime_rl.trainer.utils import TensorSummaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = Path(inspect.getfile(inspect.unwrap(accumulate_policy_provenance))).resolve()
    assert source.is_relative_to(args.source_root.resolve())
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    cases = [
        ("unequal", [7, 5, -1] if rank == 0 else [8], True, (4, 3, 1, 1, 2, 1.0, 2.0)),
        ("empty_rank_all_unknown", [-1, -1] if rank == 0 else [], True, (2, 0, 2, 0, 0, None, None)),
        ("cp_replicated", [7, 5, -1, 8, 2**63 - 1], rank == 0, (5, 4, 1, 2, 2, 1.0, 2.0)),
    ]
    results = {}
    for name, labels, observe, expected in cases:
        summaries = TensorSummaries("cpu")
        versions = torch.tensor([labels], dtype=torch.int64)
        accumulate_policy_provenance(
            summaries,
            token_policy_versions=versions,
            eligible_mask=torch.ones_like(versions, dtype=torch.bool),
            reference_version=7,
            observe=observe,
        )
        result = compute_policy_provenance(summaries)
        eligible, known, unknown, future, age_count, age_mean, age_max = expected
        for key, value in [
            ("eligible_count", eligible),
            ("known_count", known),
            ("unknown_count", unknown),
            ("future_count", future),
            ("known_fraction", known / eligible),
            ("unknown_fraction", unknown / eligible),
            ("future_fraction", future / eligible),
        ]:
            assert result[f"observed_policy_version/{key}"] == value, (name, key, result)
        assert result["sampled_weight_age/count"] == age_count
        if age_count:
            assert result["sampled_weight_age/mean"] == age_mean
            assert result["sampled_weight_age/max"] == age_max
        else:
            assert "sampled_weight_age/mean" not in result and "sampled_weight_age/max" not in result
        gathered = [None, None]
        dist.all_gather_object(gathered, result)
        assert gathered[0] == gathered[1]
        results[name] = result
    if rank == 0:
        assert not args.output.exists()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "backend": "gloo",
                    "world_size": 2,
                    "scope": "Real distributed telemetry reductions only; no GPU model or full CP-forward claim.",
                    "source": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "cases": results,
                },
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
