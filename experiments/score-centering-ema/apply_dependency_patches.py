"""Apply auditable source patches to pinned submodules, never to an installed environment."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    "verifiers": "9c6b155e10e58981a2495840b8a9026becbcfb9e",
    "renderers": "5904fa24aa73f83c1694fd85a78d0e746d468284",
}
for dependency, pin in PINS.items():
    repo = ROOT / "deps" / dependency
    patch = Path(__file__).parent / "dependency-patches" / f"{dependency}.patch"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != pin:
        raise RuntimeError(f"{dependency}: expected {pin}, found {revision}")
    applied = (
        subprocess.run(
            ["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch)], capture_output=True
        ).returncode
        == 0
    )
    if not applied:
        subprocess.run(["git", "-C", str(repo), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(repo), "apply", str(patch)], check=True)
    print(f"{dependency}: source patch present")
