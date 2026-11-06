"""Preserve bounded validation evidence before releasing the node."""

import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path

root = Path(__file__).parent
out = Path("/data/siddarth-sc-ema/evidence")
out.mkdir(exist_ok=True)
paths = []
for arm in ("sc", "ema"):
    run = Path("/tmp/sc-ema-validation") / (arm + "-8k-v2")
    paths.extend(
        p
        for p in run.rglob("*")
        if p.is_file()
        and (
            "logs" in p.parts
            or "configs" in p.parts
            or "token_exports" in p.parts
            or ("run_default" in p.parts and "rollouts" in p.parts)
            or p.name == ".metadata"
        )
    )
archive = out / "validation-outputs.tar.gz"
with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
    for p in paths:
        tar.add(p, arcname=str(p.relative_to("/tmp/sc-ema-validation")), recursive=False)
files = {
    str(p.relative_to("/tmp/sc-ema-validation")): {
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    for p in paths
}
manifest = {
    "archive": str(archive),
    "bytes": archive.stat().st_size,
    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "files": files,
    "excludes": "Full checkpoint tensors, policy weights and duplicate packed binary batches",
}
(root / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2))
# Small diagnostics suitable for Git; raw log files remain in the archive above.
for arm in ("sc", "ema"):
    run = Path("/tmp/sc-ema-validation") / (arm + "-8k-v2")
    records = []
    for p in (run / "logs/trainer").rglob("stdout.log"):
        if p.parent.name != "0":
            continue
        text = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text(errors="replace"))
        records.extend(
            line
            for line in text.splitlines()
            if any(
                x in line
                for x in (
                    "SUCCESS Step ",
                    "Research diagnostics:",
                    "Loading checkpoint",
                    "Loaded checkpoint",
                    "Peak memory:",
                    "RL trainer finished!",
                    "Fatal error",
                )
            )
        )
    (root / f"results/{arm}-8k-v2/trainer-diagnostics.txt").write_text("\n".join(records) + "\n")
for name in ("sc-audit.log", "ema-audit.log", "final-tests.log", "splits.log", "splits-disjoint.log"):
    shutil.copyfile(Path("/tmp") / name, root / "results" / name)
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}))
