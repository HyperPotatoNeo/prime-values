"""Maintain a bounded research queue using the Mac's existing official login.

Never copies credentials. API failures leave the waiting jobs unsubmitted.
Run detached under caffeinate; state and receipts are written atomically.
"""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--clusterctl", type=Path, required=True)
parser.add_argument("--state-dir", type=Path, required=True)
parser.add_argument("--once", action="store_true")
parser.add_argument("--manifest", type=Path)
parser.add_argument("--max-runs", type=int, default=6)
args = parser.parse_args()
prepared = Path(__file__).parent / "prepared"
campaign = "siddarth-sc-ema-20261106"
expanded = prepared / campaign / "expanded"
pending = [
    "siddarth-rg-ema-s11-k128-b0p03",
    "siddarth-rg-ema-s22-k128-b0p003",
    "siddarth-rg-ema-s22-k128-b0p01",
    "siddarth-rg-ema-s22-k128-b0p03",
]
if args.manifest:
    manifest = json.loads(args.manifest.read_text())
    campaign = manifest["campaign"]
    expanded = args.manifest.parent
    pending = manifest["runs"]
assert args.max_runs > 0
gpu_limit = 32 * args.max_runs
args.state_dir.mkdir(parents=True, exist_ok=True)
state_path = args.state_dir / "state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"submitted": [], "terminal": {}, "created": time.time()}


def save():
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(state_path)


def kubectl(*parts):
    for attempt in range(4):
        result = subprocess.run([str(args.clusterctl), "kubectl", "-n", "siddarth", *parts],
                                capture_output=True, text=True, timeout=90)
        if result.returncode == 0:
            return result.stdout
        # The authenticated API intermittently rejects individual requests.
        # Retrying these idempotent reads/applies does not relax admission limits.
        transient_auth = "Unauthorized" in result.stderr or "provide credentials" in result.stderr
        if not transient_auth or attempt == 3:
            raise RuntimeError(result.stderr[-1000:])
        time.sleep(1)


def apply_services(name):
    documents = (expanded / f"{name}.yaml").read_text().split("\n---\n")
    assert "kind: Job\n" in documents[-1]
    assert all("kind: Service\n" in doc for doc in documents[:-1])
    directory = args.state_dir / "services"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text("\n---\n".join(documents[:-1]))
    kubectl("apply", "-f", str(path))


def tick():
    jobs = json.loads(kubectl("get", "jobs", "-l", f"research-campaign={campaign}", "-o", "json"))["items"]
    pods = json.loads(kubectl("get", "pods", "-l", f"research-campaign={campaign}", "-o", "json"))["items"]
    names = {job["metadata"]["name"] for job in jobs}
    for name in pending:
        if name in names and name not in state["submitted"]:
            # Repair partial admission without reapplying a live Kueue Job:
            # its original suspend=true could stop the admitted pods.
            apply_services(name)
            state["submitted"].append(name)
    live = set()
    for job in jobs:
        name = job["metadata"]["name"]
        conditions = job.get("status", {}).get("conditions", [])
        terminal = [c for c in conditions if c["type"] in ("Complete", "Failed") and c["status"] == "True"]
        if terminal:
            state["terminal"][name] = {"uid": job["metadata"]["uid"], "conditions": terminal}
        else:
            live.add(name)
    receipts = []
    gpu_count = 0
    for pod in pods:
        meta, spec, status = pod["metadata"], pod["spec"], pod.get("status", {})
        count = sum(int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0)) for c in spec["containers"])
        if status.get("phase") not in ("Succeeded", "Failed"):
            gpu_count += count
            live.add(meta["labels"]["batch.kubernetes.io/job-name"])
        receipts.append({"name": meta["name"], "uid": meta["uid"], "owner": meta["labels"].get("batch.kubernetes.io/job-name"),
                         "created": meta["creationTimestamp"], "node": spec.get("nodeName"),
                         "phase": status.get("phase"), "gpus": count,
                         "containers": [{"name": c["name"], "state": c.get("state"), "restarts": c.get("restartCount")}
                                        for c in status.get("containerStatuses", [])]})
    now = time.time()
    state.update({"last_success": now, "live_jobs": sorted(live), "live_pod_requested_gpus": gpu_count,
                  "limit_jobs": args.max_runs, "limit_gpus": gpu_limit, "last_error": None})
    with (args.state_dir / "resource-receipts.jsonl").open("a") as f:
        f.write(json.dumps({"observed_at": now, "pods": receipts}) + "\n")
    if len(live) < args.max_runs and gpu_count + 32 <= gpu_limit:
        name = next((n for n in pending if n not in state["submitted"]), None)
        if name:
            apply_services(name)
            kubectl("apply", "-f", str(expanded / f"{name}.yaml"))
            state["submitted"].append(name)
            print(datetime.datetime.now(datetime.UTC).isoformat(), "SUBMITTED", name, flush=True)
    save()
    return len(state["submitted"]) == len(pending) and not live


while time.time() - state["created"] < 14 * 86400:
    try:
        done = tick()
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        # A temporary authentication/network failure must not oversubmit jobs.
        state["last_error"] = str(exc)
        state["last_error_at"] = time.time()
        save()
        print("Queue paused safely:", exc, flush=True)
        done = False
    if done or args.once:
        break
    time.sleep(30)
