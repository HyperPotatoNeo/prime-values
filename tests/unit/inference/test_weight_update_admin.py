"""The real admin client must not repeat an ambiguous collective reload."""

import asyncio
import json

import httpx
import pytest

from prime_rl.utils.client import update_weights


@pytest.mark.parametrize("outcome", ["success", "server_error", "lost_response"])
def test_reload_is_single_attempt_and_only_success_resumes(outcome):
    calls = []

    async def handle(request):
        calls.append((request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path == "/update_weights":
            if outcome == "server_error":
                return httpx.Response(500, json={"error": "reload failed"})
            if outcome == "lost_response":
                # The receiver may already have completed its collective. The
                # HTTP boundary cannot infer that replaying it would be safe.
                raise httpx.ReadError("response lost after request", request=request)
        return httpx.Response(200, json={"status": "ok"})

    async def run():
        async with httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(handle)) as client:
            await update_weights([client], weight_dir=None, step=7)

    if outcome == "success":
        asyncio.run(run())
        assert [path for path, _ in calls] == ["/pause", "/update_weights", "/resume"]
    else:
        error = httpx.HTTPStatusError if outcome == "server_error" else httpx.ReadError
        with pytest.raises(error):
            asyncio.run(run())
        assert [path for path, _ in calls] == ["/pause", "/update_weights"]
    assert calls[1][1] == {"weight_dir": None, "weight_version": 7}
