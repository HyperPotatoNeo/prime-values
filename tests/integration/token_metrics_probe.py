"""Two-rank streaming metrics against independently gathered observations.

Run with torchrun and --device cpu (Gloo) or --device cuda (NCCL).
The oracle intentionally retains test observations; production retains moments.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from prime_rl.trainer.rl.metrics import accumulate_token_metrics, compute_token_metrics, prepare_token_env_ids
from prime_rl.trainer.utils import TensorSummaries


def _data(case, rank):
    if case == "duplicate_cp":
        return {"entropy": [1.0, 3.0], "mask": [True, True], "envs": ["a", "a"], "sampled": [True, True]}
    if case == "all_empty" or (case == "empty_rank" and rank == 1):
        return {
            "entropy": [math.nan, math.inf],
            "mask": [False, False],
            "envs": ["empty", ""],
            "sampled": [False, False],
        }
    if case in {"visible_nonfinite", "nan_other_rank"}:
        rank = 1 - rank if case == "nan_other_rank" else rank
        return {
            "entropy": [math.nan, math.inf] if rank == 0 else [1.0],
            "mask": [True, True] if rank == 0 else [True],
            "envs": ["a", "a"] if rank == 0 else ["a"],
            "sampled": [False, False] if rank == 0 else [False],
        }
    if case == "opposite_infinities":
        return {"entropy": [math.inf if rank == 0 else -math.inf], "mask": [True], "envs": ["a"], "sampled": [False]}
    if rank == 0:
        return {
            "entropy": [1e12 - 2, 1e12 + 1, math.nan, 1e12 + 4, math.inf],
            "mask": [True, True, False, True, False],
            "envs": ["math", "code", "empty", "math", ""],
            "sampled": [True, False, False, True, False],
        }
    return {
        "entropy": [1e12 + 2, -math.inf, math.nan],
        "mask": [True, False, False],
        "envs": ["math", "code", "empty"],
        "sampled": [False, False, False],
    }


def _case(case, rank, device):
    data = _data(case, rank)
    summaries = TensorSummaries(device, include_std=True)
    diagnostics = TensorSummaries(device)
    diagnostics.add("explicit_empty", torch.empty(0, device=device))
    observed = {}
    # Splitting into singleton microbatches also checks local centered merges.
    for entropy_value, keep, name, sampled in zip(
        data["entropy"], data["mask"], data["envs"], data["sampled"], strict=True
    ):
        names, ids = prepare_token_env_ids([name], device)
        entropy = torch.tensor([[entropy_value]], dtype=torch.float64, device=device)
        mask = torch.tensor([[keep]], device=device)
        # Excluded CE/padding logprobs are deliberately nonfinite.
        logp = torch.tensor([[0.0 if sampled else math.nan]], device=device)
        sampling_logp = torch.tensor([[0.0 if sampled else math.inf]], device=device)
        accumulate_token_metrics(
            summaries,
            diagnostics,
            entropy=entropy,
            trainer_logprobs=logp,
            inference_logprobs=sampling_logp,
            loss_mask=mask,
            rl_weights=torch.tensor([[float(sampled)]], device=device),
            ref_kl_weights=None,
            env_names=names,
            env_ids=ids,
        )
        if keep:
            observed.setdefault("entropy/all", []).append(entropy_value)
            observed.setdefault(f"entropy/{name}", []).append(entropy_value)
        if keep and sampled:
            observed.setdefault("mismatch_kl/all", []).append(0.0)
            observed.setdefault(f"mismatch_kl/{name}", []).append(0.0)
    assert all(row.numel() == 5 and not row.requires_grad for row in summaries.summaries.values())
    gathered = [None] * 2
    dist.all_gather_object(gathered, observed)
    result = compute_token_metrics(summaries, diagnostics)
    expected_streams = {"entropy/all": []}
    for streams in gathered:
        for key, values in streams.items():
            expected_streams.setdefault(key, []).extend(values)
    expected_keys = {f"{key}/{stat}" for key in expected_streams for stat in ("mean", "std", "min", "max")}
    actual_keys = {key for key in result if key.startswith(("entropy/", "mismatch_kl/")) and "/saturated/" not in key}
    assert actual_keys == expected_keys, (case, actual_keys, expected_keys)
    for key, values in expected_streams.items():
        tensor = torch.tensor(values, dtype=torch.float64)
        expected = {
            "mean": tensor.mean().item() if values else math.nan,
            "std": tensor.std().item() if len(values) > 1 else math.nan,
            "min": tensor.min().item() if values else math.nan,
            "max": tensor.max().item() if values else math.nan,
        }
        for stat, target in expected.items():
            actual = result[f"{key}/{stat}"]
            if math.isnan(target):
                assert math.isnan(actual), (case, key, stat, actual)
            else:
                assert math.isclose(actual, target, rel_tol=1e-12, abs_tol=1e-12), (case, key, stat, actual, target)
    mismatch_count = len(expected_streams.get("mismatch_kl/all", []))
    if mismatch_count:
        assert result["mismatch_kl/all/saturated/count"] == mismatch_count
        assert result["mismatch_kl/all/saturated/mean"] == 0
    else:
        assert not any(key.startswith("mismatch_kl/") for key in result)
    assert result["explicit_empty/count"] == 0
    assert math.isnan(result["explicit_empty/mean"])
    assert summaries.compute_stats() == {}
    assert diagnostics.compute_stats() == {}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source_files = {
        "metrics": Path(inspect.getfile(inspect.unwrap(accumulate_token_metrics))).resolve(),
        "summaries": Path(inspect.getfile(TensorSummaries)).resolve(),
    }
    assert all(path.is_relative_to(args.source_root.resolve()) for path in source_files.values())
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    results = {
        case: _case(case, rank, device)
        for case in (
            "unequal",
            "empty_rank",
            "duplicate_cp",
            "all_empty",
            "visible_nonfinite",
            "nan_other_rank",
            "opposite_infinities",
        )
    }
    # A single metric makes reduction buffers contiguous views; verify global
    # reduction cannot overwrite the local count/mean needed for between-rank M2.
    single = TensorSummaries(device, include_std=True)
    single.add("one", torch.tensor([1e12 - 1 if rank == 0 else 1e12 + 1], dtype=torch.float64, device=device))
    result = single.compute_stats()
    assert result["one/count"] == 2
    assert result["one/mean"] == 1e12
    assert result["one/std"] == math.sqrt(2)
    results["single_key"] = result
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "device": str(device),
                    "world_size": 2,
                    "cases": results,
                    "source_files": {
                        key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for key, path in source_files.items()
                    },
                    "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
