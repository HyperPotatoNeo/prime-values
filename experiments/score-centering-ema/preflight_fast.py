"""Verify fresh child configs and an exhaustive allowlist against the original campaign."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import cli

p = argparse.ArgumentParser()
p.add_argument("--suffix", default="fast1")
a = p.parse_args()
root = Path(__file__).parent
prepared = root / f"prepared-{a.suffix}"
manifest = json.loads((prepared / "manifest.json").read_text())
assert len(manifest["runs"]) == 10 and manifest["simultaneous_run_limit"] == 7
for old, new in manifest["attempt_mapping"].items():
    original = json.loads((root / "prepared" / old / "resolved.json").read_text().replace(old, new))
    current = json.loads((prepared / new / "resolved.json").read_text())
    expected = copy.deepcopy(original)
    expected["trainer"]["model"]["ac_offloading"] = None
    for role in ["trainer", "orchestrator"]:
        expected[role]["wandb"]["tags"].append(a.suffix)
    assert current == expected, f"Unexpected config change: {new}"
    for role, schema in [
        ("trainer", TrainerConfig),
        ("orchestrator", OrchestratorConfig),
        ("inference", InferenceConfig),
    ]:
        actual = cli(schema, args=["@", str(prepared / new / f"{role}.json")])
        intended = schema.model_validate(current[role])
        exclude = {"deployment", "slurm", "output_dir", "dry_run"} if role == "inference" else set()
        assert actual.model_dump(exclude=exclude) == intended.model_dump(exclude=exclude), (new, role)
    assert current["trainer"]["model"]["optimization_dtype"] == "float32"
    assert current["trainer"]["model"]["reduce_dtype"] == "float32"
source = json.loads((root / "source-manifest.json").read_text())
for path, digest in source["files"].items():
    assert hashlib.sha256((root.parent.parent / path).read_bytes()).hexdigest() == digest, path
print(f"PASS: 30 child CLI configs; all original algorithm settings unchanged; {len(source['files'])} hashes")
