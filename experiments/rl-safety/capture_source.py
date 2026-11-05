"""Record exact source and dependency fingerprints alongside a completed run."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

import prime_rl.trainer.rl.loss as loss
import prime_rl.value.math as value_math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    assert Path(loss.__file__).resolve() == root / "src/prime_rl/trainer/rl/loss.py"
    assert Path(value_math.__file__).resolve() == root / "src/prime_rl/value/math.py"
    files = [root / "pyproject.toml", root / "uv.lock"]
    for pattern in ("src/**/*.py", "tests/**/*.py", "packages/**/*.py", "experiments/rl-safety/*.py"):
        files.extend(root.glob(pattern))
    report = {
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "vllm", "verifiers", "renderers", "flash-attn", "pytest")
        },
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
