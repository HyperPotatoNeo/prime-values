"""Retire consumed transport tensors while retaining lossless rollout records."""

import gzip
import shutil
from pathlib import Path


def retire_consumed_rollouts(output: Path, run_output: Path, completed_step: int, keep: int):
    if keep < 1:
        raise ValueError("Retain at least the current completed batch")
    cutoff = completed_step - keep + 1
    for root in (output / "rollouts", run_output / "rollouts"):
        for folder in root.glob("step_*"):
            suffix = folder.name[5:]
            if not folder.is_dir() or not suffix.isdigit() or int(suffix) >= cutoff:
                continue
            for file in folder.glob("*.bin"):
                file.unlink()
            for file in folder.glob("*.jsonl"):
                temporary = file.with_suffix(".jsonl.gz.tmp")
                with file.open("rb") as source, gzip.open(temporary, "wb", compresslevel=1) as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(file.with_suffix(".jsonl.gz"))
                file.unlink()
