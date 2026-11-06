"""Prepare fresh attempts, changing execution settings only; never submits jobs."""

import argparse
import json
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", default="fast1")
parser.add_argument("--replicas", type=int, default=2)
parser.add_argument("--no-recompute", action="store_true")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
root = Path(__file__).parent
old = root / "prepared"
dest = root / f"prepared-{args.suffix}"
dest.mkdir(exist_ok=True)
manifest = json.loads((old / "manifest.json").read_text())
runs = []
for name in manifest["runs"]:
    if args.smoke and "-sc-s11-" not in name:
        continue
    new = name + "-" + args.suffix
    runs.append(new)
    d = json.loads((old / name / "resolved.json").read_text().replace(name, new))
    d["trainer"]["model"]["ac_offloading"] = None
    d["trainer"]["model"]["dp_replicate"] = args.replicas
    if args.no_recompute:
        d["trainer"]["model"]["ac"] = None
    if args.smoke:
        d["max_steps"] = 5
        d["ckpt"] = None
        for role in ["trainer", "orchestrator"]:
            d[role]["max_steps"] = 5
            d[role]["ckpt"] = None
            d[role]["wandb"]["offline"] = True
    for role in ["trainer", "orchestrator"]:
        if d[role]["wandb"]:
            d[role]["wandb"]["tags"].append(args.suffix)
    folder = dest / new
    folder.mkdir(exist_ok=True)
    (folder / "resolved.json").write_text(json.dumps(d, indent=2) + "\n")
    for role in ["trainer", "orchestrator", "inference"]:
        value = d[role].copy()
        if role == "inference":
            for key in ["deployment", "slurm", "output_dir", "dry_run"]:
                value.pop(key, None)
        (folder / f"{role}.json").write_text(json.dumps(value, indent=2) + "\n")
    docs = list(
        yaml.safe_load_all(
            (old / name / "run.yaml")
            .read_text()
            .replace(name, new)
            .replace("/data/siddarth-sc-ema", "/data/siddarth-sc-ema-fast")
        )
    )
    job = docs[-1]
    campaign = "siddarth-sc-ema-" + args.suffix
    job["metadata"]["labels"]["research-campaign"] = campaign
    job["spec"]["template"].setdefault("metadata", {}).setdefault("labels", {})["research-campaign"] = campaign
    container = job["spec"]["template"]["spec"]["containers"][0]
    for env in container["env"]:
        if env["name"] == "RESEARCH_CONFIG_DIR":
            env["value"] = f"/data/siddarth-sc-ema-fast/experiments/score-centering-ema/prepared-{args.suffix}/{new}"
    for resources in ["requests", "limits"]:
        container["resources"][resources]["cpu"] = "64"
    if args.smoke:
        job["spec"]["activeDeadlineSeconds"] = 5400
    (dest / f"{new}.yaml").write_text(yaml.safe_dump_all(docs, sort_keys=False))
manifest.update(
    runs=runs,
    campaign="siddarth-sc-ema-" + args.suffix,
    simultaneous_run_limit=7,
    status="PREPARED_NOT_LAUNCHED",
    source_root="/data/siddarth-sc-ema-fast",
)
manifest.pop("lanes", None)
manifest.pop("jobs", None)
manifest["attempt_mapping"] = {
    n: n + "-" + args.suffix
    for n in json.loads((old / "manifest.json").read_text())["runs"]
    if n + "-" + args.suffix in runs
}
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(dest)
