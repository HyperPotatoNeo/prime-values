"""Exercise shutdown classification against real logs and adversarial variations."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from log_checks import FINISHED, SHUTDOWN, SHUTDOWN_COMPLETE, TRACEBACK, classify_tracebacks, marker_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_inference = (args.logs / "logs" / "inference.log").read_bytes()
    raw_launcher = (args.logs / "launcher.log").read_bytes()
    inference = re.sub(r"\x1b\[[0-9;]*m", "", raw_inference.decode())
    launcher = re.sub(r"\x1b\[[0-9;]*m", "", raw_launcher.decode())
    shutdown_line = next(line for line in inference.splitlines() if SHUTDOWN in line)
    late_shutdown = inference.replace(shutdown_line, "") + "\n" + shutdown_line
    later = (marker_time(inference, SHUTDOWN) + 1) % (24 * 3600)
    later_clock = f"{later // 3600:02d}:{later // 60 % 60:02d}:{later % 60:02d}"
    late_launcher = "\n".join(
        re.sub(r"\b\d{2}:\d{2}:\d{2}\b", later_clock, line) if FINISHED in line else line
        for line in launcher.splitlines()
    )
    cases = [
        ("actual_completed_shutdown", "inference.log", inference, launcher, True, "expected_shutdown_engine_dead"),
        ("missing_launcher_completion", "inference.log", inference, launcher.replace(FINISHED, ""), True, "rejected"),
        ("shutdown_marker_after_trace", "inference.log", late_shutdown, launcher, True, "rejected"),
        (
            "different_exception",
            "inference.log",
            inference.replace("EngineDeadError", "RuntimeError"),
            launcher,
            True,
            "rejected",
        ),
        (
            "environment_startup_failure",
            "envs/train/reverse-text-v1.log",
            TRACEBACK + "\nImportError: failed",
            launcher,
            True,
            "rejected",
        ),
        ("launcher_finished_after_shutdown", "inference.log", inference, late_launcher, True, "rejected"),
        (
            "incomplete_shutdown_final",
            "inference.log",
            inference.replace(SHUTDOWN_COMPLETE, ""),
            launcher,
            True,
            "rejected",
        ),
        (
            "incomplete_shutdown_live",
            "inference.log",
            inference.replace(SHUTDOWN_COMPLETE, ""),
            launcher,
            False,
            "shutdown_pending",
        ),
        (
            "additional_exception",
            "inference.log",
            inference + "\n" + TRACEBACK + "\nValueError: unrelated",
            launcher,
            True,
            "rejected",
        ),
        (
            "engine_error_after_completed_shutdown",
            "inference.log",
            inference + "\n" + TRACEBACK + "\nEngineDeadError: late failure",
            launcher,
            True,
            "rejected",
        ),
        (
            "completion_before_terminal_exception",
            "inference.log",
            inference.replace(SHUTDOWN_COMPLETE, "").replace(
                "vllm.v1.engine.exceptions.EngineDeadError:",
                SHUTDOWN_COMPLETE + "\nvllm.v1.engine.exceptions.EngineDeadError:",
            ),
            launcher,
            True,
            "rejected",
        ),
    ]
    results = []
    for name, path, content, launch_log, final, expected in cases:
        try:
            actual = classify_tracebacks(path, content, launch_log, final=final)
        except RuntimeError:
            actual = "rejected"
        results.append({"case": name, "expected": expected, "actual": actual, "passed": actual == expected})
    output = {
        "input_sha256": {
            "inference.log": hashlib.sha256(raw_inference).hexdigest(),
            "launcher.log": hashlib.sha256(raw_launcher).hexdigest(),
        },
        "controls": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    assert all(result["passed"] for result in results), output
    print(f"All {len(results)} shutdown-classification controls passed")


if __name__ == "__main__":
    main()
