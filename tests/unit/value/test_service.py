import pytest

import prime_rl.value.service as service_module
from prime_rl.configs.value import ValueEvaluatorConfig
from prime_rl.value.service import (
    ValueRequestService,
    ValueRequestTimeout,
    ValueRequestTooLarge,
    ValueServiceUnavailable,
)
from prime_rl.value.types import ValueEvaluationRequest


def _service(
    *,
    max_batch_tokens: int = 10,
    max_pending_requests: int = 8,
    max_pending_tokens: int = 100,
    request_timeout: float = 1.0,
    batch_wait_ms: float = 0,
) -> ValueRequestService:
    config = ValueEvaluatorConfig(
        max_batch_tokens=max_batch_tokens,
        batch_wait_ms=batch_wait_ms,
        max_pending_requests=max_pending_requests,
        max_pending_tokens=max_pending_tokens,
        request_timeout=request_timeout,
    )
    return ValueRequestService(config, seq_len=16, vocab_size=100, version=0)


def _request(*sequences: tuple[int, int]) -> ValueEvaluationRequest:
    return ValueEvaluationRequest(token_ids=[[token] * length for length, token in sequences])


def _complete(service: ValueRequestService, batch) -> None:
    service.complete(batch, [[0.0] * len(tokens) for tokens in batch.token_ids], version=0)


def test_fifo_batching_does_not_skip_a_blocked_head_request():
    service = _service()
    service._submit(_request((6, 1)))
    service._submit(_request((6, 2)))
    service._submit(_request((2, 3)))

    first = service.take_batch(wait_for_first=False)
    second = service.take_batch(wait_for_first=False)

    assert first is not None and first.token_ids == [[1] * 6]
    assert second is not None and second.token_ids == [[2] * 6, [3] * 2]
    _complete(service, first)
    _complete(service, second)


def test_completion_restores_logical_request_boundaries_and_releases_capacity():
    service = _service()
    first = service._submit(_request((2, 1), (1, 2)))
    second = service._submit(_request((3, 3)))
    batch = service.take_batch(wait_for_first=False)
    assert batch is not None

    service.complete(batch, [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6]], version=7)

    assert not service.cancel(first)
    assert first.future.result().values == [[0.1, 0.2], [0.3]]
    assert first.future.result().version == 7
    assert second.future.result().values == [[0.4, 0.5, 0.6]]
    assert second.future.result().version == 7
    metrics = service.metrics()
    assert metrics["value/service_pending_requests"] == 0
    assert metrics["value/service_pending_tokens"] == 0
    assert metrics["value/service_completed"] == 2


def test_request_above_batch_ceiling_runs_alone_but_admission_limit_is_enforced():
    service = _service(max_pending_tokens=20)
    service._submit(_request((6, 1), (6, 2)))
    service._submit(_request((2, 3)))

    oversized_batch = service.take_batch(wait_for_first=False)
    following_batch = service.take_batch(wait_for_first=False)

    assert oversized_batch is not None and oversized_batch.token_ids == [[1] * 6, [2] * 6]
    assert following_batch is not None and following_batch.token_ids == [[3] * 2]
    with pytest.raises(ValueRequestTooLarge, match="21 tokens"):
        service._submit(_request((11, 4), (10, 5)))
    _complete(service, oversized_batch)
    _complete(service, following_batch)


def test_capacity_counts_running_work_and_cancel_releases_only_queued_work():
    service = _service(max_pending_requests=2, max_pending_tokens=10)
    running_ticket = service._submit(_request((6, 1)))
    running_batch = service.take_batch(wait_for_first=False)
    assert running_batch is not None

    assert service.cancel(running_ticket)
    queued_ticket = service._submit(_request((4, 2)))
    with pytest.raises(ValueServiceUnavailable, match="queue is full"):
        service._submit(_request((1, 3)))

    service.cancel(queued_ticket)
    replacement = service._submit(_request((4, 4)))
    metrics = service.metrics()
    assert metrics["value/service_pending_requests"] == 2
    assert metrics["value/service_pending_tokens"] == 10
    assert metrics["value/service_abandoned"] == 1

    _complete(service, running_batch)
    service.cancel(replacement)
    metrics = service.metrics()
    assert metrics["value/service_pending_requests"] == 0
    assert metrics["value/service_pending_tokens"] == 0
    assert metrics["value/service_abandoned"] == 1
    assert metrics["value/service_completed"] == 0
    assert not running_ticket.future.done()


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        ([], "at least one sequence"),
        ([[]], "length must be"),
        ([[1] * 17], "length must be"),
        ([[100]], "outside the model vocabulary"),
    ],
)
def test_invalid_requests_are_rejected_before_admission(token_ids, message):
    service = _service()

    with pytest.raises(ValueError, match=message):
        service._submit(ValueEvaluationRequest(token_ids=token_ids))

    assert service.metrics()["value/service_admitted"] == 0


def test_close_fails_queued_work_but_allows_selected_work_to_finish():
    service = _service()
    service._submit(_request((6, 1)))
    running_batch = service.take_batch(wait_for_first=False)
    assert running_batch is not None
    queued_ticket = service._submit(_request((2, 2)))

    service.close()

    assert isinstance(queued_ticket.future.exception(), ValueServiceUnavailable)
    assert not running_batch.tickets[0].future.done()
    assert service.metrics()["value/service_pending_requests"] == 1
    with pytest.raises(ValueServiceUnavailable, match="unavailable"):
        service._submit(_request((1, 3)))

    _complete(service, running_batch)
    assert service.metrics()["value/service_pending_requests"] == 0


def test_submit_timeout_cancels_queued_ticket_without_sleeping(monkeypatch):
    service = _service(request_timeout=1.0)
    calls = 0

    def monotonic() -> float:
        nonlocal calls
        calls += 1
        return 10.0 if calls == 1 else 12.0

    monkeypatch.setattr(service_module.time, "monotonic", monotonic)

    with pytest.raises(ValueRequestTimeout, match="timed out"):
        service.submit_and_wait(_request((2, 1)))

    metrics = service.metrics()
    assert metrics["value/service_pending_requests"] == 0
    assert metrics["value/service_pending_tokens"] == 0
    assert metrics["value/service_expired"] == 1
    assert metrics["value/service_abandoned"] == 0


def test_batch_collection_never_waits_past_the_oldest_request_deadline(monkeypatch):
    service = _service(request_timeout=1.0, batch_wait_ms=2_000)
    clock = {"now": 0.0}
    waits: list[float] = []
    monkeypatch.setattr(service_module.time, "monotonic", lambda: clock["now"])

    def wait(timeout: float) -> None:
        waits.append(timeout)
        clock["now"] += timeout

    monkeypatch.setattr(service._condition, "wait", wait)
    service._submit(_request((2, 1)))

    batch = service.take_batch(wait_for_first=False)

    assert batch is not None
    assert waits == [1.0]
    _complete(service, batch)


def test_request_arriving_after_collection_deadline_stays_at_fifo_head(monkeypatch):
    service = _service()
    service.config.batch_wait_ms = 1_000
    clock = {"now": 0.0}
    submitted_late = False
    monkeypatch.setattr(service_module.time, "monotonic", lambda: clock["now"])

    def wait(timeout: float) -> None:
        nonlocal submitted_late
        clock["now"] += timeout
        if not submitted_late:
            submitted_late = True
            clock["now"] += 1.0
            service._submit(_request((2, 2)))

    monkeypatch.setattr(service._condition, "wait", wait)
    service._submit(_request((2, 1)))

    first = service.take_batch(wait_for_first=False)
    second = service.take_batch(wait_for_first=False)

    assert first is not None and first.token_ids == [[1, 1]]
    assert second is not None and second.token_ids == [[2, 2]]
    _complete(service, first)
    _complete(service, second)
