"""Profile actual old accumulation block versus new streaming helper, without a model."""

import argparse
import hashlib
import inspect
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from prime_rl.trainer.rl.loss import DIAGNOSTIC_RATIO_LIMIT, compute_mismatch_kl
from prime_rl.trainer.rl.metrics import accumulate_token_metrics, prepare_token_env_ids
from prime_rl.trainer.utils import TensorSummaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    helper_source = Path(inspect.getfile(inspect.unwrap(accumulate_token_metrics))).resolve()
    summaries_source = Path(inspect.getfile(TensorSummaries)).resolve()
    assert helper_source.is_relative_to(args.source_root.resolve())
    assert summaries_source.is_relative_to(args.source_root.resolve())
    frozen_source = args.frozen_source_root / "src/prime_rl/trainer/rl/train.py"
    old_source = frozen_source.read_text()
    start = old_source.index('            entropy = out["entropy"][loss_mask].detach().to("cpu")')
    end = old_source.index("            token_exporter.export(", start)
    # Execute the actual frozen trainer's metric accumulation source, excluding
    # exporter, losses and debug formatting. No mirrored baseline implementation.
    old_block = compile(textwrap.dedent(old_source[start:end]), str(frozen_source), "exec")
    tokens, repetitions = 65536, 3
    entropy = torch.linspace(0.1, 10.0, tokens, device="cuda").reshape(1, -1)
    logp = -entropy / 3
    sampling_logp = -entropy / 4
    mask = (torch.arange(tokens, device="cuda") % 3 != 0).reshape(1, -1)
    env_names = [f"env_{i % 4}" for i in range(tokens)]
    loss = entropy.sum() * 0
    records = {}
    for mode in ("frozen", "streaming"):
        raw = defaultdict(list)
        summaries = TensorSummaries(include_std=True)
        diagnostics = TensorSummaries()
        scope = {
            "torch": torch,
            "out": {"entropy": entropy, "logprobs": logp},
            "inference_logprobs": sampling_logp,
            "loss_mask": mask,
            "loss": loss,
            "micro_batch": {"env_names": env_names},
            "rl_weights": None,
            "ref_kl_weights": None,
            "tensors": raw,
            "diagnostic_summaries": diagnostics,
            "compute_mismatch_kl": compute_mismatch_kl,
            "DIAGNOSTIC_RATIO_LIMIT": DIAGNOSTIC_RATIO_LIMIT,
        }

        def operation():
            if mode == "frozen":
                exec(old_block, scope)
            else:
                names, ids = prepare_token_env_ids(env_names, entropy.device)
                accumulate_token_metrics(
                    summaries,
                    diagnostics,
                    entropy=entropy,
                    trainer_logprobs=logp,
                    inference_logprobs=sampling_logp,
                    loss_mask=mask,
                    rl_weights=None,
                    ref_kl_weights=None,
                    env_names=names,
                    env_ids=ids,
                )
                # This scalar copy remains in the production caller unchanged.
                raw["loss"].append(loss.detach().to("cpu").unsqueeze(0))

        operation()
        raw.clear()
        summaries.summaries.clear()
        diagnostics.summaries.clear()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(repetitions):
                with record_function("token_metric_accumulation"):
                    operation()
        torch.cuda.synchronize()
        counts = {event.key: event.count for event in prof.key_averages()}
        trace = args.output / f"{mode}.trace.json"
        prof.export_chrome_trace(str(trace))
        copies = [
            {"name": event.get("name"), "args": event.get("args", {})}
            for event in json.loads(trace.read_text())["traceEvents"]
            if "memcpy" in event.get("name", "").lower()
        ]
        records[mode] = {
            "event_counts": counts,
            "copy_and_sync_events": {
                key: value for key, value in counts.items() if any(w in key.lower() for w in ("copy", "memcpy", "sync"))
            },
            "cuda_copy_events": copies,
            "retained_host_tensor_elements": sum(t.numel() for values in raw.values() for t in values),
            "retained_device_summary_elements": sum(t.numel() for t in summaries.summaries.values())
            + sum(t.numel() for t in diagnostics.summaries.values()),
        }
    assert records["frozen"]["retained_host_tensor_elements"] > tokens
    assert records["streaming"]["retained_host_tensor_elements"] == repetitions
    assert records["streaming"]["retained_device_summary_elements"] == 54
    assert records["frozen"]["event_counts"].get("aten::nonzero", 0) > 0
    assert records["streaming"]["event_counts"].get("aten::nonzero", 0) == 0
    sources = {"frozen_trainer": frozen_source, "metrics": helper_source, "summaries": summaries_source}
    receipt = {
        "scope": "Accumulation only, including new environment-ID upload and unchanged scalar-loss CPU copy; no end-step reductions, model, debug formatting, or throughput claim.",
        "limitations": "Input masks are ordinary finite observations here; independent two-rank tests cover numeric semantics. Scalar loss CPU copies and other unchanged trainer paths still synchronize. Profiler reports actual copy/sync events; no all-sync-free claim.",
        "tokens": tokens,
        "repetitions": repetitions,
        "profiles": records,
        "source_files": {
            key: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in sources.items()
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
