"""Run real distributed checks serially, with bounded fatal-rank controls."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []

    def run(name, script, ranks=2, extra=(), env=None, expected=0):
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=5",
            "45s",
            "uv",
            "run",
            "--no-sync",
            "torchrun",
            "--standalone",
            f"--nproc-per-node={ranks}",
            script,
            *extra,
        ]
        start = time.monotonic()
        log = args.output / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        entry = {"name": name, "returncode": result.returncode, "elapsed_seconds": time.monotonic() - start}
        results.append(entry)
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        assert result.returncode == expected, entry
        if expected:
            assert "intentional rank failure with unmatched collective" in log.read_text(), log

    if args.baseline_source:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(args.baseline_source.resolve()), str(Path("packages/prime-rl-configs/src").resolve())]
        )
        subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-c",
                "import inspect; from prime_rl.utils.utils import clean_exit; "
                "assert 'os._exit' not in inspect.getsource(clean_exit)",
            ],
            env=env,
            check=True,
        )
        run("baseline-fatal", "tests/integration/distributed_failure_probe.py", env=env, expected=124)
    run("fixed-fatal", "tests/integration/distributed_failure_probe.py", expected=1)
    run("normal-shutdown", "tests/integration/distributed_failure_probe.py", extra=["--success"])
    for ranks in (1, 2):
        run(f"policy-normalization-{ranks}", "tests/integration/policy_loss_scale_probe.py", ranks=ranks)


if __name__ == "__main__":
    main()
