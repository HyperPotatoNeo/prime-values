"""Build a fixed RG-Mix dataset for reuse across runs.

Generates a mixture of the five reasoning-gym task variants used by RG-Mix and
saves it in the on-disk format ``rg_mix.load_environment`` expects:

    <out>/dataset/          # HF dataset with {question, answer=global_index} rows
    <out>/metadata.json     # {entry_map: [[variant_id, local_idx], ...], entries_cache: {idx: entry}}

Generation is parallelized across processes (some variants, e.g. zebra_puzzles_7,
are very slow single-threaded). Each (variant, chunk) is generated with a
distinct deterministic seed, then results are concatenated in a fixed order and
the rows are shuffled (seeded) so variants are interleaved. The environment
slices the first ``num_train_examples`` rows for training and the next
``num_eval_examples`` for eval, so build a total >= train + eval.

Usage:
    uv run --extra rgmix python environments/rg_mix/build_dataset.py --out <dir> --total 10500
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import Dataset
from rg_mix import TASK_VARIANTS


def _gen_chunk(task: str, config: dict, chunk_seed: int, size: int) -> list[dict]:
    import reasoning_gym as rg

    ds = rg.create_dataset(task, seed=chunk_seed, size=size, **config)
    return [dict(ds[i]) for i in range(len(ds))]


def build_dataset(total: int, out_dir: str, seed: int, workers: int, chunk_size: int) -> None:
    n_variants = len(TASK_VARIANTS)
    per_variant = -(-total // n_variants)  # ceil division

    # One task per (variant, chunk) with a distinct deterministic seed.
    jobs = []  # (variant_index, chunk_index, task, config, chunk_seed, size)
    for vi, variant in enumerate(TASK_VARIANTS):
        remaining, ci = per_variant, 0
        while remaining > 0:
            size = min(chunk_size, remaining)
            jobs.append((vi, ci, variant["task"], variant["config"], seed + vi * 100_000 + ci, size))
            remaining -= size
            ci += 1

    results: dict[tuple[int, int], list[dict]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_gen_chunk, t, cfg, s, sz): (vi, ci) for (vi, ci, t, cfg, s, sz) in jobs}
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  generated {done}/{len(jobs)} chunks", flush=True)

    # Concatenate deterministically: variant by variant, chunk by chunk.
    entry_map: list[list] = []
    entries_cache: dict[int, dict] = {}
    rows: list[dict] = []
    global_idx = 0
    for vi, variant in enumerate(TASK_VARIANTS):
        ci = 0
        while (vi, ci) in results:
            for entry in results[(vi, ci)]:
                entry_map.append([variant["id"], global_idx])
                entries_cache[global_idx] = entry
                rows.append({"question": entry["question"], "answer": global_idx})
                global_idx += 1
            ci += 1

    random.Random(seed).shuffle(rows)

    os.makedirs(out_dir, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(os.path.join(out_dir, "dataset"))
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"entry_map": entry_map, "entries_cache": entries_cache}, f)

    print(f"Wrote {len(rows)} RG-Mix examples to {out_dir} ({per_variant}/variant x {n_variants} variants)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed RG-Mix dataset.")
    parser.add_argument("--out", type=str, required=True, help="Output dataset root directory.")
    parser.add_argument("--total", type=int, default=10500, help="Total examples to generate (train + eval).")
    parser.add_argument("--seed", type=int, default=42, help="Generation + shuffle seed.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel generation processes.")
    parser.add_argument("--chunk-size", type=int, default=25, help="Examples generated per parallel task.")
    args = parser.parse_args()
    build_dataset(args.total, args.out, args.seed, args.workers, args.chunk_size)


if __name__ == "__main__":
    main()
