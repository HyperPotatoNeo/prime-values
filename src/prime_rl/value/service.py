from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import msgspec

from prime_rl.configs.value import ValueEvaluatorConfig
from prime_rl.utils.logger import get_logger
from prime_rl.value.types import ValueEvaluationRequest, ValueEvaluationResponse, ValueVersionResponse

MAX_VALUE_REQUEST_BYTES = 16 * 1024 * 1024


class ValueServiceUnavailable(RuntimeError):
    pass


class ValueRequestTooLarge(ValueError):
    pass


class ValueRequestTimeout(TimeoutError):
    pass


@dataclass(eq=False)
class _RequestTicket:
    request: ValueEvaluationRequest
    tokens: int
    admitted_at: float
    deadline: float
    future: Future[ValueEvaluationResponse] = field(default_factory=Future)
    abandoned: bool = False
    released: bool = False


@dataclass(frozen=True)
class ValueRequestBatch:
    tickets: list[_RequestTicket]
    token_ids: list[list[int]]


class ValueRequestService:
    """Bounded FIFO request plane shared by dedicated and trainer serving.

    A disconnected HTTP client may leave its ticket queued until its deadline;
    the trusted job-network API deliberately does not probe client sockets.
    """

    def __init__(
        self,
        config: ValueEvaluatorConfig,
        *,
        seq_len: int,
        vocab_size: int,
        version: int | None = None,
    ):
        self.config = config
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self._condition = threading.Condition()
        self._queue: deque[_RequestTicket] = deque()
        self._version = version
        self._error: BaseException | None = None
        self._closed = False
        self._pending_requests = 0
        self._pending_tokens = 0
        self._metrics = {
            "admitted": 0,
            "rejected_full": 0,
            "expired": 0,
            "abandoned": 0,
            "completed": 0,
            "failed": 0,
        }
        self._max_wait = 0.0

    def set_version(self, version: int) -> None:
        with self._condition:
            self._version = version

    def fail_service(self, error: BaseException) -> None:
        with self._condition:
            self._error = error
        self.close(error)

    def snapshot(self) -> tuple[int | None, BaseException | None, bool]:
        with self._condition:
            return self._version, self._error, self._closed

    def submit_and_wait(self, request: ValueEvaluationRequest) -> ValueEvaluationResponse:
        ticket = self._submit(request)
        try:
            return ticket.future.result(timeout=max(ticket.deadline - time.monotonic(), 0.0))
        except FutureTimeoutError:
            if not self.cancel(ticket):
                return ticket.future.result()
            raise ValueRequestTimeout("value evaluation request timed out") from None

    def _submit(self, request: ValueEvaluationRequest) -> _RequestTicket:
        self._validate(request)
        tokens = sum(map(len, request.token_ids))
        if tokens > self.config.max_pending_tokens:
            raise ValueRequestTooLarge(
                f"value request has {tokens} tokens; maximum is {self.config.max_pending_tokens}"
            )

        admitted_at = time.monotonic()
        ticket = _RequestTicket(
            request=request,
            tokens=tokens,
            admitted_at=admitted_at,
            deadline=admitted_at + self.config.request_timeout,
        )
        with self._condition:
            if self._closed or self._error is not None or self._version is None:
                raise ValueServiceUnavailable("value evaluator is unavailable")
            if (
                self._pending_requests >= self.config.max_pending_requests
                or self._pending_tokens + tokens > self.config.max_pending_tokens
            ):
                self._metrics["rejected_full"] += 1
                raise ValueServiceUnavailable("value evaluator request queue is full")
            self._queue.append(ticket)
            self._pending_requests += 1
            self._pending_tokens += tokens
            self._metrics["admitted"] += 1
            self._condition.notify_all()
        return ticket

    def _validate(self, request: ValueEvaluationRequest) -> None:
        if not request.token_ids:
            raise ValueError("value evaluation request must contain at least one sequence")
        for index, tokens in enumerate(request.token_ids):
            if not tokens or len(tokens) > self.seq_len:
                raise ValueError(f"value sequence {index} length must be in [1, {self.seq_len}]")
            if any(token < 0 or token >= self.vocab_size for token in tokens):
                raise ValueError(f"value sequence {index} contains a token outside the model vocabulary")

    def has_queued(self) -> bool:
        with self._condition:
            self._expire_head(time.monotonic())
            return bool(self._queue)

    def wait_for_work(self, timeout: float) -> None:
        with self._condition:
            if not self._queue and not self._closed:
                self._condition.wait(timeout)

    def take_batch(self, *, wait_for_first: bool) -> ValueRequestBatch | None:
        with self._condition:
            while True:
                self._expire_head(time.monotonic())
                while not self._queue:
                    if self._closed or not wait_for_first:
                        return None
                    self._condition.wait()
                    self._expire_head(time.monotonic())
                first = self._take_head()
                if first is not None:
                    break
            tickets = [first]
            tokens = first.tokens
            batch_deadline = min(first.admitted_at + self.config.batch_wait_ms / 1000.0, first.deadline)
            waited_for_arrival = False

            while tokens < self.config.max_batch_tokens:
                self._expire_head(time.monotonic())
                if self._queue:
                    next_ticket = self._queue[0]
                    if waited_for_arrival and next_ticket.admitted_at > batch_deadline:
                        break
                    if tokens + next_ticket.tokens > self.config.max_batch_tokens:
                        break
                    next_ticket = self._take_head()
                    if next_ticket is not None:
                        tickets.append(next_ticket)
                        tokens += next_ticket.tokens
                    continue

                remaining = batch_deadline - time.monotonic()
                if remaining <= 0 or self._closed:
                    break
                waited_for_arrival = True
                self._condition.wait(remaining)

            wait = time.monotonic() - first.admitted_at
            self._max_wait = max(self._max_wait, wait)
            return ValueRequestBatch(
                tickets=tickets,
                token_ids=[sequence for ticket in tickets for sequence in ticket.request.token_ids],
            )

    def _take_head(self) -> _RequestTicket | None:
        ticket = self._queue.popleft()
        if ticket.future.set_running_or_notify_cancel():
            return ticket
        self._release(ticket)
        return None

    def _expire_head(self, now: float) -> None:
        while self._queue and self._queue[0].deadline <= now:
            ticket = self._queue.popleft()
            self._release(ticket)
            self._metrics["expired"] += 1
            if not ticket.future.done():
                ticket.future.set_exception(ValueRequestTimeout("value evaluation request timed out"))

    def cancel(self, ticket: _RequestTicket) -> bool:
        with self._condition:
            if ticket.released:
                return False
            if ticket.abandoned:
                return True
            if ticket.future.cancel():
                self._queue.remove(ticket)
                self._release(ticket)
                self._metrics["expired"] += 1
            else:
                ticket.abandoned = True
                self._metrics["abandoned"] += 1
            self._condition.notify_all()
            return True

    def complete(self, batch: ValueRequestBatch, values: list[list[float]], version: int) -> None:
        expected = sum(len(ticket.request.token_ids) for ticket in batch.tickets)
        if len(values) != expected:
            raise RuntimeError(f"value response has {len(values)} sequences for {expected} requested sequences")

        offset = 0
        for ticket in batch.tickets:
            count = len(ticket.request.token_ids)
            response = ValueEvaluationResponse(values=values[offset : offset + count], version=version)
            offset += count
            with self._condition:
                self._release(ticket)
                if not ticket.abandoned:
                    ticket.future.set_result(response)
                    self._metrics["completed"] += 1

    def fail(self, batch: ValueRequestBatch, error: BaseException) -> None:
        for ticket in batch.tickets:
            with self._condition:
                self._release(ticket)
                if not ticket.abandoned:
                    ticket.future.set_exception(error)
                    self._metrics["failed"] += 1

    def close(self, error: BaseException | None = None) -> None:
        error = error or ValueServiceUnavailable("value evaluator is stopping")
        with self._condition:
            if self._closed:
                return
            self._closed = True
            tickets = list(self._queue)
            self._queue.clear()
            for ticket in tickets:
                self._release(ticket)
            self._condition.notify_all()
        for ticket in tickets:
            if not ticket.future.done():
                ticket.future.set_exception(error)

    def _release(self, ticket: _RequestTicket) -> None:
        if ticket.released:
            return
        ticket.released = True
        self._pending_requests -= 1
        self._pending_tokens -= ticket.tokens

    def metrics(self) -> dict[str, float]:
        with self._condition:
            oldest_wait = time.monotonic() - self._queue[0].admitted_at if self._queue else 0.0
            return {
                "value/service_pending_requests": float(self._pending_requests),
                "value/service_pending_tokens": float(self._pending_tokens),
                "value/service_oldest_wait_seconds": oldest_wait,
                "value/service_max_wait_seconds": self._max_wait,
                **{f"value/service_{name}": float(value) for name, value in self._metrics.items()},
            }


class ValueHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: ValueRequestService):
        self.service = service
        super().__init__(address, ValueRequestHandler)

    def server_close(self) -> None:
        self.service.close()
        super().server_close()


class ValueRequestHandler(BaseHTTPRequestHandler):
    request_decoder = msgspec.msgpack.Decoder(type=ValueEvaluationRequest)
    encoder = msgspec.msgpack.Encoder()

    @property
    def value_server(self) -> ValueHTTPServer:
        return cast(ValueHTTPServer, self.server)

    def log_message(self, format: str, *args) -> None:
        get_logger().debug(format % args)

    def do_GET(self) -> None:
        version, error, closed = self.value_server.service.snapshot()
        if self.path == "/health":
            if error is not None or closed or version is None:
                self._write(HTTPStatus.SERVICE_UNAVAILABLE, b"value evaluator unavailable", "text/plain")
            else:
                self._write(HTTPStatus.OK, b"ok", "text/plain")
            return
        if self.path == "/version":
            if error is not None or closed or version is None:
                self._write(HTTPStatus.SERVICE_UNAVAILABLE, b"value evaluator unavailable", "text/plain")
            else:
                self._write(HTTPStatus.OK, self.encoder.encode(ValueVersionResponse(version=version)))
            return
        self._write(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/evaluate":
            self._write(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return

        try:
            raw_length = self.headers.get("content-length")
            if raw_length is None:
                raise ValueError("missing content-length")
            length = int(raw_length)
            if length <= 0:
                raise ValueError("content-length must be positive")
            if length > MAX_VALUE_REQUEST_BYTES:
                raise ValueRequestTooLarge(f"value request body exceeds {MAX_VALUE_REQUEST_BYTES} bytes")
            request = self.request_decoder.decode(self.rfile.read(length))
            response = self.value_server.service.submit_and_wait(request)
            status, payload, content_type = HTTPStatus.OK, self.encoder.encode(response), "application/msgpack"
        except ValueRequestTooLarge as error:
            status, payload, content_type = HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(error).encode(), "text/plain"
        except ValueRequestTimeout as error:
            status, payload, content_type = HTTPStatus.GATEWAY_TIMEOUT, str(error).encode(), "text/plain"
        except ValueServiceUnavailable as error:
            status, payload, content_type = HTTPStatus.SERVICE_UNAVAILABLE, str(error).encode(), "text/plain"
        except (ValueError, msgspec.DecodeError) as error:
            status, payload, content_type = HTTPStatus.BAD_REQUEST, str(error).encode(), "text/plain"
        except Exception as error:
            get_logger().exception("Value evaluation failed")
            status, payload, content_type = HTTPStatus.INTERNAL_SERVER_ERROR, str(error).encode(), "text/plain"
        self._write(status, payload, content_type)

    def _write(self, status: HTTPStatus, payload: bytes, content_type: str = "application/msgpack") -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
