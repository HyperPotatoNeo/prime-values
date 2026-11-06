#!/usr/bin/env python3
"""Pre-generate the RG-Mix dataset and save to disk.

Generates the weighted reasoning-gym mix dataset and saves it so subsequent
runs can load instantly, avoiding the slow puzzle generation each time.

Uses multiprocessing to parallelize entry generation across CPU cores.

Usage:
    python generate_rg_mix_dataset.py --output /path/to/output \
        --num-train 7500 --num-test 100 --seed 42
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

from datasets import Dataset


# Must be at module level for multiprocessing pickling
_WORKER_DATASETS = {}


def _init_worker(task_configs, seed, total_examples):
    """Initialize each worker with its own rg dataset instances."""
    import reasoning_gym as rg
    global _WORKER_DATASETS
    for i, (vid, task_name, config) in enumerate(task_configs):
        ds = rg.create_dataset(
            task_name,
            seed=seed + i + 1,
            size=total_examples,
            **config,
        )
        _WORKER_DATASETS[vid] = ds


def _generate_entry(args):
    """Generate a single entry in a worker process."""
    vid, idx = args
    ds = _WORKER_DATASETS[vid]
    entry = ds[idx]
    return (vid, idx, entry)


def main():
    parser = argparse.ArgumentParser(description="Pre-generate RG-Mix dataset")
    parser.add_argument("--output", "-o", required=True, help="Output directory path")
    parser.add_argument("--num-train", type=int, default=7500, help="Number of training examples")
    parser.add_argument("--num-test", type=int, default=100, help="Number of test/eval examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes (default: cpu_count)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    from rg_mix_env import TASK_VARIANTS

    total_examples = args.num_train + args.num_test
    seed = args.seed
    n_workers = args.workers or min(cpu_count(), 32)

    print(f"Generating RG-Mix dataset: {args.num_train} train + {args.num_test} test (seed={seed})")
    print(f"Total combined examples: {total_examples}")
    print(f"Using {n_workers} workers", flush=True)

    # Phase 1: Pre-compute which entries we need via weighted sampling
    print("\nPre-computing weighted sample assignments...", flush=True)
    rng = random.Random(seed)
    weights = [v["weight"] for v in TASK_VARIANTS]

    # assignments[i] = (variant_id, entry_idx) for each global index
    assignments = []
    needed_entries = []  # (vid, entry_idx) to generate
    for i in range(total_examples):
        chosen_idx = rng.choices(range(len(TASK_VARIANTS)), weights=weights, k=1)[0]
        variant = TASK_VARIANTS[chosen_idx]
        vid = variant["id"]
        entry_idx = i % total_examples  # dataset size == total_examples
        assignments.append((vid, entry_idx))
        needed_entries.append((vid, entry_idx))

    task_counts = Counter(vid for vid, _ in assignments)
    print("Entries to generate per task:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        pct = count / total_examples * 100
        print(f"  {task}: {count} ({pct:.1f}%)")

    # Phase 2: Generate entries in parallel
    print(f"\nGenerating {len(needed_entries)} entries with {n_workers} workers...", flush=True)
    t0 = time.time()

    task_configs = [
        (v["id"], v["task"], v["config"]) for v in TASK_VARIANTS
    ]

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(task_configs, seed, total_examples),
    ) as pool:
        results = []
        for i, result in enumerate(pool.imap(
            _generate_entry, needed_entries, chunksize=16
        )):
            results.append(result)
            if (i + 1) % 500 == 0 or i == len(needed_entries) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(needed_entries) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(needed_entries)}] {elapsed:.0f}s elapsed, "
                      f"{rate:.1f} entries/s, ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed:.1f}s ({len(results)/elapsed:.1f} entries/s)", flush=True)

    # Phase 3: Build the dataset rows
    print("\nBuilding dataset...", flush=True)
    entry_map = []
    entries_cache = {}
    rows = []

    for i, (vid, entry_idx, entry) in enumerate(results):
        entry_map.append((vid, entry_idx))
        entries_cache[i] = entry
        rows.append({
            "question": entry["question"],
            "answer": str(i),
            "task": vid,
        })

    # Print final distribution
    final_counts = Counter(r["task"] for r in rows)
    print("\nFinal task distribution:")
    for task, count in sorted(final_counts.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100
        print(f"  {task}: {count} ({pct:.1f}%)")

    # Phase 4: Save
    print(f"\nSaving to {output_path} ...", flush=True)
    hf_ds = Dataset.from_list(rows)
    hf_ds.save_to_disk(str(output_path / "dataset"))
    print("  Saved HF dataset", flush=True)

    metadata = {
        "entry_map": entry_map,
        "entries_cache": {str(k): v for k, v in entries_cache.items()},
        "num_train": args.num_train,
        "num_test": args.num_test,
        "seed": seed,
        "total_examples": len(rows),
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    meta_size_mb = (output_path / "metadata.json").stat().st_size / 1024 / 1024
    print(f"  Saved metadata.json ({meta_size_mb:.1f} MB)", flush=True)
    print("Done!")


if __name__ == "__main__":
    main()
