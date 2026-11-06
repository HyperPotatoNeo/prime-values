"""Read campaign state and step-indexed metrics from the shared volume."""

import argparse
import json
import re
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
args = parser.parse_args()
prepared = Path(__file__).parent / "prepared"
manifest = json.loads((prepared / "manifest.json").read_text())
result = {"observed_at": time.time(), "campaign": manifest["campaign"], "runs": {}}
for name in manifest["runs"]:
    out = Path("/data/outputs") / name
    run = {"status": "queued", "metrics": {}, "wandb": [], "net_ib": False, "errors": []}
    if out.exists():
        run["status"] = "started"
    if (out / "policy.done").exists():
        run["status"] = "complete"
    if (out / "failure.txt").exists():
        run["status"] = "failed"
        run["errors"].append((out / "failure.txt").read_text())
    for f in (out / "logs").glob("*-metrics.jsonl"):
        lines = f.read_text().splitlines()
        if lines:
            run["metrics"][f.name] = json.loads(lines[-1])
    for f in (out / "logs").glob("*.log"):
        text = re.sub(r"\x1b\[[0-9;]*m", "", f.read_text(errors="replace"))
        run["wandb"].extend(re.findall(r"https://wandb.ai/swish/siddarth-prime-values/runs/[a-z0-9]+", text))
        if "NET/IB" in text:
            run["net_ib"] = True
        run["errors"].extend(line for line in text.splitlines()[-100:] if "Fatal error" in line or "Traceback" in line)
    run["wandb"] = sorted(set(run["wandb"]))
    result["runs"][name] = run
if args.output:
    args.output.write_text(json.dumps(result, indent=2) + "\n")
for name, run in result["runs"].items():
    print(name, run["status"], "NET/IB", run["net_ib"], "errors", run["errors"])
    print(" ".join(run["wandb"]))
    for component, metrics in run["metrics"].items():
        print(component, {k: v for k, v in metrics.items()
                          if k in ("step", "time") or re.search(r"reward/mean|loss/mean|kl/mean|kl/max|off_policy|policy_age|grad_norm", k)})
