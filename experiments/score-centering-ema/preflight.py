"""Check effective child CLI configs and the complete paired-seed comparison."""

import copy
import hashlib
import json
from pathlib import Path

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import cli


def strip_capture(value):
    if isinstance(value, dict):
        if value.get("extra_args") == {"score_centering_top_k": 128}:
            value.pop("extra_args")
        for item in value.values():
            strip_capture(item)
    elif isinstance(value, list):
        for item in value:
            strip_capture(item)


root = Path(__file__).parent
prepared = root / "prepared"
manifest = json.loads((prepared / "manifest.json").read_text())
matched = {}
for name in manifest["runs"]:
    c = json.loads((prepared / name / "resolved.json").read_text())
    for role, schema in [("trainer", TrainerConfig), ("orchestrator", OrchestratorConfig), ("inference", InferenceConfig)]:
        actual = cli(schema, args=["@", str(prepared / name / f"{role}.toml")])
        exclude = {"deployment", "slurm", "output_dir", "dry_run"} if role == "inference" else set()
        expected = schema.model_validate(copy.deepcopy(c[role]))
        assert actual.model_dump(mode="json", exclude=exclude) == expected.model_dump(mode="json", exclude=exclude), (name, role)
    normalized = json.loads(json.dumps(c).replace(name, "RUN"))
    normalized["trainer"]["loss"].pop("score_centering")
    normalized["trainer"]["loss"].pop("ema_kl_beta")
    strip_capture(normalized)
    for role in ["trainer", "orchestrator"]:
        normalized[role]["wandb"].pop("tags")
    seed = c["trainer"]["seed"]
    assert c["orchestrator"]["max_off_policy_steps"] == 20
    if seed in matched:
        assert normalized == matched[seed], name
    else:
        matched[seed] = normalized
assert len(manifest["runs"]) == 10 and len(set(sum(manifest["lanes"], []))) == 10
source = json.loads((root / "source-manifest.json").read_text())
repo = root.parent.parent
for path, digest in source["files"].items():
    assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == digest, path
print(f"PASS: 30 CLI roundtrips, 10 matched configs, off-policy limit20, {len(source['files'])} source hashes")
