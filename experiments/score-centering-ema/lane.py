"""Run a fixed sequence on four reserved nodes without cluster API credentials."""

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/siddarth-sc-ema")
PREPARED = ROOT / "experiments/score-centering-ema/prepared"
campaign = os.environ["RESEARCH_CAMPAIGN"]
lane = int(os.environ["RESEARCH_LANE"])
role = int(os.environ["JOB_COMPLETION_INDEX"])
manifest = json.loads((PREPARED / "manifest.json").read_text())
assert campaign == manifest["campaign"]
state = Path("/data/outputs") / campaign / f"lane{lane}"
state.mkdir(parents=True, exist_ok=True)
failure = state / "failure.txt"
child = None


def event(kind, **fields):
    with (state / f"role{role}.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.time(), "event": kind, **fields}) + "\n")


def barrier(folder, phase):
    (folder / f"{phase}-{role}").touch()
    deadline = time.monotonic() + 1800
    while not all((folder / f"{phase}-{r}").exists() for r in range(4)):
        if failure.exists():
            raise RuntimeError(failure.read_text())
        if time.monotonic() > deadline:
            raise TimeoutError(f"Four-node barrier timed out: {phase}")
        time.sleep(3)


def interrupted(signum, frame):
    raise SystemExit(f"Received signal {signum}")


signal.signal(signal.SIGTERM, interrupted)

try:
    for name in manifest["lanes"][lane]:
        if failure.exists():
            raise RuntimeError(failure.read_text())
        folder = state / name
        folder.mkdir(exist_ok=True)
        out = Path("/data/outputs") / name
        if role == 2:
            # Never silently resume or overwrite an earlier attempt.
            out.mkdir(exist_ok=False)
            free = shutil.disk_usage(out).free
            if free < 200 * 2**30:
                raise RuntimeError(f"Less than 200 GiB free before {name}: {free}")
        barrier(folder, "ready")
        resolved = json.loads((PREPARED / name / "resolved.json").read_text())
        env = dict(os.environ, RESEARCH_RUN_NAME=name,
                   RESEARCH_SEED=str(resolved["trainer"]["seed"]),
                   RESEARCH_CONFIG_DIR=str(PREPARED / name))
        event("start", run=name)
        child = subprocess.Popen(["uv", "run", "--no-sync", "python",
                                  "experiments/score-centering-ema/launch.py"], cwd=ROOT, env=env)
        deadline = time.monotonic() + 72 * 3600
        while child.poll() is None:
            if failure.exists():
                raise RuntimeError(failure.read_text())
            if time.monotonic() > deadline:
                raise TimeoutError(f"Run exceeded 72 hours: {name}")
            time.sleep(3)
        if child.returncode:
            raise RuntimeError(f"{name} role {role} exited {child.returncode}")
        event("complete", run=name)
        barrier(folder, "complete")
    event("lane_complete")
except BaseException as exc:
    failure.write_text(f"role {role}: {exc}\n")
    event("failed", error=str(exc))
    raise
finally:
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=50)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
