"""Only retire published weights after all inference engines acknowledge adoption."""

import shutil
from pathlib import Path


def acknowledge(directory: Path, step: int):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "ADOPTED.tmp"
    temporary.write_text(str(step))
    temporary.replace(directory / "ADOPTED")


def clean_acknowledged(directory: Path, published_step: int):
    marker = directory / "ADOPTED"
    if not marker.exists():
        return
    adopted = int(marker.read_text())
    if adopted > published_step:
        return  # A rollback resume has not yet replaced the old acknowledgement.
    for candidate in directory.glob("step_*"):
        if candidate.is_dir() and candidate.name[5:].isdigit() and int(candidate.name[5:]) < adopted:
            shutil.rmtree(candidate)
