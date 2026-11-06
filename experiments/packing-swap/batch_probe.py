"""Run the actual batch packer and loss-count inputs on one mixed-weight batch."""

import argparse
import hashlib
import inspect
import json
import random
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import torch

from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.rl.normalization import count_loss_tokens
from prime_rl.trainer.utils import balanced_partition, build_bin_cost
from prime_rl.transport.types import RoutedExperts, TrainingSample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    sources = {
        name: Path(inspect.getfile(obj)).resolve()
        for name, obj in {
            "batch": prepare_batch,
            "partition": balanced_partition,
            "normalization": count_loss_tokens,
            "transport": TrainingSample,
        }.items()
    }
    assert all(path.is_relative_to(args.source_root.resolve()) for path in sources.values())
    rng = random.Random(1701)
    samples = []
    for index in range(128):
        length = rng.randrange(20, 180)
        mask = [position >= length // 2 for position in range(length)]
        routes = np.arange(length * 4, dtype=np.uint8).reshape(length, 2, 2)
        samples.append(
            TrainingSample(
                token_ids=[index * 200 + position for position in range(length)],
                mask=mask,
                logprobs=[-0.2 - position / 1000 for position in range(length)],
                temperatures=[1.0] * length,
                advantages=[(position % 5 - 2) / 10 for position in range(length)],
                env_name=f"env-{index % 3}",
                ref_logprobs=[-0.25] * length,
                rl_weights=[float(on and index % 3 != 0) for on in mask],
                ce_weights=[0.3 if on and index % 3 == 0 else 0.0 for on in mask],
                ref_kl_weights=[0.5 if on and index % 5 == 0 else 0.0 for on in mask],
                routed_experts=RoutedExperts(data=routes.tobytes(), shape=list(routes.shape), dtype="uint8"),
            )
        )
    model = SimpleNamespace(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=32000,
        intermediate_size=512,
        num_hidden_layers=4,
        head_dim=16,
    )
    packed = prepare_batch(samples, 256, 4, [0] * len(samples), 1, build_bin_cost(model), pad_to_multiple_of=16)
    encoded = msgspec.msgpack.encode(packed)
    counts = []
    for batches in packed:
        tensor_batches = [
            {
                field: torch.tensor(getattr(batch, field)) if getattr(batch, field) is not None else None
                for field in ("loss_mask", "rl_weights", "ce_weights", "ref_kl_weights")
            }
            for batch in batches
        ]
        counts.append(count_loss_tokens(tensor_batches))
    global_counts = [sum(rank[component] for rank in counts) for component in range(3)]
    assert all(count > 0 for count in global_counts)
    args.output.mkdir(parents=True)
    (args.output / "packed.msgpack").write_bytes(encoded)
    receipt = {
        "packed_sha256": hashlib.sha256(encoded).hexdigest(),
        "samples": len(samples),
        "micro_batches_per_rank": [len(batches) for batches in packed],
        "per_rank_component_counts": counts,
        "global_component_counts": global_counts,
        "source_files": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in sources.items()
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Actual prepare_batch, transport serialization and count_loss_tokens; global count is the sum over four simulated data ranks. No distributed collective, model or GPU run.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("packed_sha256", "global_component_counts")}))


if __name__ == "__main__":
    main()
