from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass

import msgspec
import zmq
import zmq.asyncio

from prime_rl.configs.value import ZMQValueTransportConfig
from prime_rl.utils.logger import get_logger
from prime_rl.value.types import ValueTrainingRollout

_NONBLOCKING_RESPONSE_WAIT_MS = 2


class _RolloutRequest(msgspec.Struct, array_like=True, gc=False):
    request_id: int
    limit: int


class _RolloutResponse(msgspec.Struct, array_like=True, gc=False):
    request_id: int
    count: int


@dataclass(frozen=True)
class _EncodedRollout:
    payload: bytes
    num_tokens: int


class _DropOldestRolloutQueue:
    """Bounded FIFO of encoded rollouts waiting for the value trainer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: deque[_EncodedRollout] = deque()
        self.num_bytes = 0
        self.num_tokens = 0

    def __len__(self) -> int:
        return len(self._items)

    def append(self, item: _EncodedRollout) -> _EncodedRollout | None:
        dropped = self.popleft() if len(self._items) == self.capacity else None
        self._items.append(item)
        self.num_bytes += len(item.payload)
        self.num_tokens += item.num_tokens
        return dropped

    def popleft(self) -> _EncodedRollout:
        item = self._items.popleft()
        self.num_bytes -= len(item.payload)
        self.num_tokens -= item.num_tokens
        return item

    def restore_oldest(self, items: list[_EncodedRollout]) -> int:
        """Restore a failed reply, dropping its oldest items if newer arrivals filled the queue."""
        keep = min(len(items), self.capacity - len(self._items))
        for item in reversed(items[-keep:] if keep else []):
            self._items.appendleft(item)
            self.num_bytes += len(item.payload)
            self.num_tokens += item.num_tokens
        return len(items) - keep

    def clear(self) -> None:
        self._items.clear()
        self.num_bytes = 0
        self.num_tokens = 0


class ValueRolloutPublisher:
    """Nonblocking producer queue for finalized value-training rollouts.

    ``publish`` only serializes and enqueues. A background task owns the ZMQ
    socket and may wait indefinitely for the trainer without delaying rollout
    processing. When the waiting queue is full, the oldest waiting rollout is
    replaced by the new one.
    """

    def __init__(self, config: ZMQValueTransportConfig) -> None:
        self.encoder = msgspec.msgpack.Encoder()
        self.request_decoder = msgspec.msgpack.Decoder(type=_RolloutRequest)
        self.endpoint = f"tcp://{config.host}:{config.port}"
        self._queue = _DropOldestRolloutQueue(config.max_pending_rollouts)
        self._wake = asyncio.Event()
        self._socket: zmq.asyncio.Socket | None = None
        self._responder_task: asyncio.Task | None = None
        self._closed = False
        self.enqueued = 0
        self.sent = 0
        self.dropped_oldest = 0
        self.responder_failures = 0

    @property
    def capacity(self) -> int:
        return self._queue.capacity

    @property
    def pending_rollouts(self) -> int:
        return len(self._queue)

    async def start(self) -> None:
        if self._responder_task is not None:
            raise RuntimeError("value rollout publisher is already started")
        if self._closed:
            raise RuntimeError("value rollout publisher is closed")
        self._socket = zmq.asyncio.Context.instance().socket(zmq.REP)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.IMMEDIATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        self._responder_task = asyncio.create_task(self._respond_loop(), name="value-rollout-responder")
        self._responder_task.add_done_callback(self._on_responder_done)
        get_logger().info(
            f"Value rollout responder connected to {self.endpoint} (max_pending_rollouts={self.capacity})"
        )

    def publish(self, rollout: ValueTrainingRollout) -> None:
        if self._responder_task is None:
            raise RuntimeError("value rollout publisher is not started")
        if self._closed:
            raise RuntimeError("value rollout publisher is closed")
        item = _EncodedRollout(
            payload=self.encoder.encode(rollout),
            num_tokens=sum(len(sample.token_ids) for sample in rollout.samples),
        )
        dropped = self._queue.append(item)
        self.enqueued += 1
        if dropped is not None:
            self.dropped_oldest += 1
        self._wake.set()

    def metrics(self) -> dict[str, float]:
        return {
            "value/rollout_queue_enqueued": float(self.enqueued),
            "value/rollout_queue_sent": float(self.sent),
            "value/rollout_queue_dropped_oldest": float(self.dropped_oldest),
            "value/rollout_queue_drop_rate": self.dropped_oldest / self.enqueued if self.enqueued else 0.0,
            "value/rollout_queue_capacity": float(self.capacity),
            "value/rollout_queue_pending": float(self.pending_rollouts),
            "value/rollout_queue_pending_bytes": float(self._queue.num_bytes),
            "value/rollout_queue_pending_tokens": float(self._queue.num_tokens),
            "value/rollout_responder_failures": float(self.responder_failures),
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._responder_task is not None:
            self._responder_task.cancel()
            try:
                await self._responder_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass  # Non-cancellation failures are reported by _on_responder_done.
            self._responder_task = None
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._queue.clear()

    async def _respond_loop(self) -> None:
        assert self._socket is not None
        while True:
            request_payload = await self._socket.recv(copy=False)
            request = self.request_decoder.decode(request_payload)
            if request.limit < 1:
                raise RuntimeError(f"value rollout request limit must be positive, got {request.limit}")

            while not self._queue:
                self._wake.clear()
                if not self._queue:
                    await self._wake.wait()

            while True:
                await self._socket.poll(flags=zmq.POLLOUT)
                items = [self._queue.popleft() for _ in range(min(request.limit, len(self._queue)))]
                response = self.encoder.encode(_RolloutResponse(request_id=request.request_id, count=len(items)))
                try:
                    send = self._socket.send_multipart(
                        [response, *(item.payload for item in items)], flags=zmq.DONTWAIT, copy=False
                    )
                    await send
                except zmq.Again:
                    self.dropped_oldest += self._queue.restore_oldest(items)
                    continue
                self.sent += len(items)
                break

    def _on_responder_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.responder_failures += 1
            get_logger().error(f"Value rollout responder failed; rollout processing will continue: {error!r}")


class ValueRolloutReceiver:
    """Pull bounded FIFO slices from the producer when replay can admit them."""

    def __init__(self, config: ZMQValueTransportConfig) -> None:
        self.encoder = msgspec.msgpack.Encoder()
        self.response_decoder = msgspec.msgpack.Decoder(type=_RolloutResponse)
        self.rollout_decoder = msgspec.msgpack.Decoder(type=ValueTrainingRollout)
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.poll_timeout_ms = config.poll_timeout_ms
        self.endpoint = f"tcp://{config.bind_host}:{config.port}"
        self.socket.bind(self.endpoint)
        self._next_request_id = 0
        self._pending_request: _RolloutRequest | None = None
        self._buffered_rollouts: deque[ValueTrainingRollout] = deque()
        get_logger().info(f"Value trainer listening for rollout pull responses on {self.endpoint}")

    def receive_available(
        self,
        limit: int,
        *,
        wait_for_first: bool = False,
    ) -> list[ValueTrainingRollout]:
        """Request up to ``limit`` FIFO rollouts within one bounded response wait."""
        if limit < 0:
            raise ValueError("receive limit cannot be negative")
        if limit == 0:
            return []
        if self._buffered_rollouts:
            return self._take_buffered(limit)

        timeout_ms = self.poll_timeout_ms if wait_for_first else _NONBLOCKING_RESPONSE_WAIT_MS
        deadline = time.monotonic() + timeout_ms / 1000
        if self._pending_request is None:
            if not self._poll_until(zmq.POLLOUT, deadline):
                return []
            request = _RolloutRequest(request_id=self._next_request_id, limit=limit)
            try:
                self.socket.send(self.encoder.encode(request), flags=zmq.DONTWAIT, copy=False)
            except zmq.Again:
                return []
            self._pending_request = request
            self._next_request_id += 1

        if not self._poll_until(zmq.POLLIN, deadline):
            return []
        try:
            frames = self.socket.recv_multipart(flags=zmq.DONTWAIT, copy=False)
        except zmq.Again:
            return []
        return self._accept_response(frames, limit)

    def _accept_response(self, frames: list[zmq.Frame], limit: int) -> list[ValueTrainingRollout]:
        assert self._pending_request is not None
        if not frames:
            raise RuntimeError("value rollout response is empty")
        response = self.response_decoder.decode(frames[0])
        if response.request_id != self._pending_request.request_id:
            raise RuntimeError(
                f"value rollout response id {response.request_id} does not match "
                f"request {self._pending_request.request_id}"
            )
        if response.count != len(frames) - 1:
            raise RuntimeError(f"value rollout response declared {response.count} rollout(s), got {len(frames) - 1}")
        if response.count > self._pending_request.limit:
            raise RuntimeError(
                f"value rollout response exceeded request limit {response.count}>{self._pending_request.limit}"
            )
        self._buffered_rollouts.extend(self.rollout_decoder.decode(frame) for frame in frames[1:])
        self._pending_request = None
        return self._take_buffered(limit)

    def _poll_until(self, event: int, deadline: float) -> bool:
        remaining_ms = max(int((deadline - time.monotonic()) * 1000), 0)
        return bool(self.socket.poll(timeout=remaining_ms, flags=event) & event)

    def _take_buffered(self, limit: int) -> list[ValueTrainingRollout]:
        return [self._buffered_rollouts.popleft() for _ in range(min(limit, len(self._buffered_rollouts)))]

    def close(self) -> None:
        self._pending_request = None
        self._buffered_rollouts.clear()
        self.socket.close(linger=0)
