"""Two real Gloo ranks: unequal populations, empty rank and CP observation ownership."""

import argparse
import hashlib
import inspect
import json
import math
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
    reducer_source = Path(inspect.getfile(TensorSummaries)).resolve()
    assert reducer_source.is_relative_to(args.source_root.resolve())
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
            mismatch_kl=torch.tensor([[float(i + 1) for i in range(len(labels))]]),
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
        expected_mismatch = {
            "unequal": {"0": [1], "2-3": [2], "unknown": [3], "future": [1]},
            "empty_rank_all_unknown": {"unknown": [1, 2]},
            "cp_replicated": {"0": [1], "2-3": [2], "unknown": [3], "future": [4, 5]},
        }[name]
        for population, values in expected_mismatch.items():
            key = f"mismatch_kl_by_sampled_age/{population}"
            assert result[f"{key}/count"] == len(values)
            assert result[f"{key}/mean"] == sum(values) / len(values)
            assert result[f"{key}/max"] == max(values)
            assert result[f"{key}/saturated_fraction"] == 0
        assert (
            sum(
                value
                for key, value in result.items()
                if key.startswith("mismatch_kl_by_sampled_age/") and key.endswith("/count")
            )
            == eligible
        )
        gathered = [None, None]
        dist.all_gather_object(gathered, result)
        assert gathered[0] == gathered[1]
        results[name] = result
    # NaN on either rank must remain visible, while a masked NaN cannot contaminate
    # another bucket. Positive infinity is a saturated value, not a NaN proxy.
    for nan_rank in (0, 1):
        summaries = TensorSummaries("cpu")
        accumulate_policy_provenance(
            summaries,
            token_policy_versions=torch.tensor([[7, 6, 5]]),
            eligible_mask=torch.tensor([[True, False, True]]),
            reference_version=7,
            mismatch_kl=torch.tensor(
                [[float("nan") if rank == nan_rank else 2, float("nan"), float("inf") if rank == 0 else 3]]
            ),
        )
        result = compute_policy_provenance(summaries)
        assert result["mismatch_kl_by_sampled_age/0/count"] == 2
        assert math.isnan(result["mismatch_kl_by_sampled_age/0/mean"])
        assert math.isnan(result["mismatch_kl_by_sampled_age/0/max"])
        assert result["mismatch_kl_by_sampled_age/0/saturated_fraction"] == 0
        assert result["mismatch_kl_by_sampled_age/1/count"] == 0
        assert "mismatch_kl_by_sampled_age/1/mean" not in result
        assert result["mismatch_kl_by_sampled_age/2-3/count"] == 2
        assert result["mismatch_kl_by_sampled_age/2-3/mean"] == float("inf")
        assert result["mismatch_kl_by_sampled_age/2-3/max"] == float("inf")
        assert result["mismatch_kl_by_sampled_age/2-3/saturated_fraction"] == 0.5
        results[f"nan_rank_{nan_rank}"] = result
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
                    "reducer_source": str(reducer_source),
                    "reducer_sha256": hashlib.sha256(reducer_source.read_bytes()).hexdigest(),
                    "torch_version": torch.__version__,
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
