"""Freeze disjoint validation/test manifests without running any inference."""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset
from rg_mix_env import _generate_dataset, _load_dataset


def prompt_hash(row):
    return hashlib.sha256(row["question"].encode()).hexdigest()


def main(train_path, validation_path, report):
    rows, _, _, _ = _load_dataset(train_path)
    assert len(rows) >= 10256
    train_hashes = {prompt_hash(r) for r in rows[:10000]}
    test_hashes = {prompt_hash(r) for r in rows[10000:10256]}
    if train_hashes & test_hashes:
        raise ValueError("Existing train/test prompts overlap")
    if validation_path.exists():
        valid, _, _, _ = _load_dataset(validation_path)
    else:
        # One unused training row keeps the legacy environment schema populated.
        candidates, raw_mapping, raw_entries, _ = _generate_dataset(1, 512, 43)
        seen = train_hashes | test_hashes
        selected = []
        for i, row in enumerate(candidates):
            h = prompt_hash(row)
            if h not in seen:
                selected.append(i)
                seen.add(h)
            if len(selected) == 257:
                break
        if len(selected) != 257:
            raise ValueError("Not enough unique disjoint validation candidates")
        valid = [{**candidates[i], "answer": str(j)} for j, i in enumerate(selected)]
        mapping = [raw_mapping[i] for i in selected]
        entries = {j: raw_entries[i] for j, i in enumerate(selected)}
        validation_path.mkdir(parents=True)
        Dataset.from_list(valid).save_to_disk(str(validation_path / "dataset"))
        (validation_path / "metadata.json").write_text(
            json.dumps(
                {
                    "entry_map": mapping,
                    "entries_cache": entries,
                    "seed": 43,
                    "num_train_examples": 1,
                    "num_eval_examples": 256,
                    "candidate_indices": selected,
                    "selection": "first 257 unique prompts disjoint from fixed train and test; row 0 unused",
                }
            )
        )
    assert len(valid) == 257
    assert len({prompt_hash(r) for r in valid}) == 257
    valid_hashes = {prompt_hash(r) for r in valid[1:]}
    if valid_hashes & (train_hashes | test_hashes):
        raise ValueError("Validation prompts overlap training/test; freeze a revised split before tuning")
    payload = {
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "training": {"seed": 42, "n": 10000},
        "test": {
            "seed": 42,
            "n": 256,
            "answer_ids": [r["answer"] for r in rows[10000:10256]],
            "prompt_sha256": [prompt_hash(r) for r in rows[10000:10256]],
        },
        "validation": {
            "seed": 43,
            "n": 256,
            "answer_ids": [r["answer"] for r in valid[1:]],
            "prompt_sha256": [prompt_hash(r) for r in valid[1:]],
        },
        "cross_split_prompt_overlap": 0,
    }
    report.write_text(json.dumps(payload, indent=2))
    print("Frozen 256 validation and 256 test prompts; no cross-split prompt overlap")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", type=Path, default=Path("/data/rg-dataset"))
    p.add_argument("--validation-path", type=Path, default=Path("/data/rg-validation-s43-disjoint"))
    p.add_argument("--report", type=Path, default=Path(__file__).parent / "splits.json")
    a = p.parse_args()
    main(a.train_path, a.validation_path, a.report)
