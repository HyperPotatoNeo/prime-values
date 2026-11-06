"""Measure boundary preparation only; this is not a training throughput benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import prime_rl.utils.sequence as sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    args.output.mkdir(parents=True, exist_ok=True)
    # Deliberately uneven documents and a distinct final padding segment.
    lengths = [1023, 513, 2047, 1025, 511, 1537, 1021, 515]
    total = sum(lengths)
    positions = torch.tensor([[position for length in lengths for position in range(length)]], device="cuda")

    def old_boundary():
        return sequence.get_cu_seqlens_from_position_ids(positions)

    def explicit_boundary():
        metadata = sequence.prepare_packed_sequence_metadata(lengths, total_tokens=total, device=positions.device)
        return metadata.cu_seqlens, metadata.max_seqlen

    old_cu, old_max = old_boundary()
    new_cu, new_max = explicit_boundary()
    torch.testing.assert_close(old_cu, new_cu, rtol=0, atol=0)
    assert old_max == new_max
    receipts = {}
    for name, operation in [("position", old_boundary), ("explicit", explicit_boundary)]:
        for _ in range(3):
            operation()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                with record_function("boundary_preparation"):
                    operation()
        torch.cuda.synchronize()
        counts = {event.key: event.count for event in prof.key_averages()}
        prof.export_chrome_trace(str(args.output / f"{name}.trace.json"))
        receipts[name] = {
            "nonzero": counts.get("aten::nonzero", 0),
            "local_scalar_dense": counts.get("aten::_local_scalar_dense", 0),
            "item": counts.get("aten::item", 0),
            "copy_and_sync_events": {
                key: count
                for key, count in counts.items()
                if any(word in key.lower() for word in ("copy", "memcpy", "sync"))
            },
            "event_counts": counts,
        }
    assert receipts["position"]["nonzero"] >= 10
    assert receipts["position"]["local_scalar_dense"] >= 10
    assert receipts["explicit"]["nonzero"] == 0
    assert receipts["explicit"]["local_scalar_dense"] == 0
    assert receipts["explicit"]["item"] == 0
    source = Path(sequence.__file__).resolve()
    result = {
        "scope": "boundary preparation only; no full-model or throughput claim",
        "limitation": "The host-to-device offset upload may still synchronize. This checks removal of device-derived boundary discovery, not all synchronization.",
        "lengths": lengths,
        "tokens": total,
        "repetitions": 10,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "exact_offsets_and_max": True,
        "profiles": receipts,
    }
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
