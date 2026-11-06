"""Recognize the specific vLLM shutdown race without tolerating training failures."""

import re

TRACEBACK = "Traceback (most recent call last):"
SHUTDOWN = "[shutdown] API server: shutdown triggered"
SHUTDOWN_COMPLETE = "Application shutdown complete."
FINISHED = "Training finished!"
EXCEPTION = re.compile(r"(?:^|\s)([\w.]+(?:Error|Exception)|KeyboardInterrupt|SystemExit)(?::[^\n]*)?$", re.MULTILINE)


def marker_time(content, marker):
    line = next(line for line in content.splitlines() if marker in line)
    match = re.search(r"\b(\d{2}):(\d{2}):(\d{2})\b", line)
    if match is None:
        raise RuntimeError(f"No timestamp for shutdown ordering marker: {marker}")
    hour, minute, second = map(int, match.groups())
    return hour * 3600 + minute * 60 + second


def classify_tracebacks(relative_path, content, launcher, *, final):
    if TRACEBACK not in content:
        return "clear"
    if relative_path != "inference.log" or FINISHED not in launcher:
        raise RuntimeError(f"Unexpected component traceback: {relative_path}")
    first_trace = content.index(TRACEBACK)
    shutdown = content.find(SHUTDOWN)
    if shutdown < 0 or shutdown > first_trace:
        raise RuntimeError("Inference traceback preceded its shutdown marker")
    elapsed = (marker_time(content, SHUTDOWN) - marker_time(launcher, FINISHED)) % (24 * 3600)
    if elapsed > 1800:
        raise RuntimeError("Inference shutdown was not within 30 minutes after launcher training completion")
    terminal_matches = list(EXCEPTION.finditer(content, first_trace))
    last_terminal = max((match.end() for match in terminal_matches), default=first_trace)
    complete = content.find(SHUTDOWN_COMPLETE, max(content.rindex(TRACEBACK), last_terminal))
    incomplete = False
    for block in content[first_trace:].split(TRACEBACK)[1:]:
        exceptions = [name.rsplit(".", 1)[-1] for name in EXCEPTION.findall(block)]
        if any(name != "EngineDeadError" for name in exceptions):
            raise RuntimeError(f"Unexpected inference shutdown exception: {exceptions}")
        if not exceptions:
            incomplete = True
    if final and (incomplete or complete < 0):
        raise RuntimeError("Inference shutdown traceback did not resolve to completed EngineDeadError cleanup")
    if incomplete or complete < 0:
        return "shutdown_pending"
    return "expected_shutdown_engine_dead"
