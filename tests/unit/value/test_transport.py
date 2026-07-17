import asyncio
import socket

import msgspec
import pytest
import zmq

from prime_rl.configs.value import ZMQValueTransportConfig
from prime_rl.value.transport import (
    ValueRolloutPublisher,
    ValueRolloutReceiver,
    _DropOldestRolloutQueue,
    _EncodedRollout,
    _RolloutRequest,
    _RolloutResponse,
)
from prime_rl.value.types import ValueTrainingRollout, ValueTrainingSample


def _item(payload: bytes, num_tokens: int) -> _EncodedRollout:
    return _EncodedRollout(payload=payload, num_tokens=num_tokens)


def _rollout(rollout_id: int) -> ValueTrainingRollout:
    return ValueTrainingRollout(
        samples=[ValueTrainingSample(token_ids=[rollout_id], mask=[True], targets=[0.0])],
        rollout_id=rollout_id,
        policy_version=0,
        value_version=0,
    )


def _transport_config(*, max_pending_rollouts: int = 10) -> ZMQValueTransportConfig:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return ZMQValueTransportConfig(
        host="127.0.0.1",
        bind_host="127.0.0.1",
        port=port,
        poll_timeout_ms=20,
        max_pending_rollouts=max_pending_rollouts,
    )


async def _receive_response(receiver: ValueRolloutReceiver, limit: int) -> list[ValueTrainingRollout]:
    deadline = asyncio.get_running_loop().time() + 1
    while asyncio.get_running_loop().time() < deadline:
        rollouts = receiver.receive_available(limit)
        if rollouts:
            return rollouts
        await asyncio.sleep(0.005)
    raise AssertionError("timed out waiting for value rollout response")


class _BlockedReplySocket:
    def __init__(self, request: _RolloutRequest) -> None:
        self.request = msgspec.msgpack.encode(request)
        self.poll_started = asyncio.Event()
        self.writable = asyncio.Event()
        self.sent = asyncio.Event()
        self.frames: list[bytes] = []
        self.closed = False

    async def recv(self, *, copy: bool):
        return self.request

    async def poll(self, *, flags: int) -> int:
        self.poll_started.set()
        await self.writable.wait()
        return flags

    def send_multipart(self, frames, *, flags: int, copy: bool):
        assert flags == zmq.DONTWAIT
        self.frames = list(frames)
        self.sent.set()
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return future

    def close(self, *, linger: int) -> None:
        self.closed = True


class _AgainOnceReplySocket(_BlockedReplySocket):
    def __init__(self, request: _RolloutRequest) -> None:
        super().__init__(request)
        self.writable.set()
        self.first_send = asyncio.Event()
        self.release_first_send: asyncio.Future | None = None

    def send_multipart(self, frames, *, flags: int, copy: bool):
        if self.release_first_send is None:
            self.first_send.set()
            self.release_first_send = asyncio.get_running_loop().create_future()
            return self.release_first_send
        return super().send_multipart(frames, flags=flags, copy=copy)


class _ResponseSocket:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames

    def poll(self, *, timeout: int, flags: int) -> int:
        return flags

    def send(self, *_args, **_kwargs) -> None:
        pass

    def recv_multipart(self, *_args, **_kwargs):
        return self.frames

    def close(self, *, linger: int) -> None:
        pass


def test_rollout_queue_drops_oldest_and_preserves_fifo_order():
    queue = _DropOldestRolloutQueue(capacity=2)
    first = _item(b"a", 1)
    second = _item(b"bb", 2)
    newest = _item(b"ccc", 3)

    assert queue.append(first) is None
    assert queue.append(second) is None
    assert queue.append(newest) is first

    assert queue.popleft() is second
    assert queue.popleft() is newest


def test_rollout_queue_tracks_pending_bytes_and_tokens():
    queue = _DropOldestRolloutQueue(capacity=2)
    queue.append(_item(b"aa", 3))
    queue.append(_item(b"bbb", 5))

    assert len(queue) == 2
    assert queue.num_bytes == 5
    assert queue.num_tokens == 8

    queue.append(_item(b"c", 7))
    assert queue.num_bytes == 4
    assert queue.num_tokens == 12

    queue.clear()
    assert len(queue) == 0
    assert queue.num_bytes == 0
    assert queue.num_tokens == 0


def test_receiver_rejects_negative_limit_without_receiving():
    receiver = ValueRolloutReceiver(_transport_config())

    try:
        with pytest.raises(ValueError, match="cannot be negative"):
            receiver.receive_available(-1)

        assert receiver._pending_request is None
    finally:
        receiver.close()


def test_receiver_without_a_peer_does_not_leave_a_request_pending():
    receiver = ValueRolloutReceiver(_transport_config())

    try:
        assert receiver.receive_available(2) == []
        assert receiver._pending_request is None
    finally:
        receiver.close()


def test_publisher_enqueue_is_bounded_without_waiting_for_trainer():
    async def run_test() -> None:
        publisher = ValueRolloutPublisher(ZMQValueTransportConfig(max_pending_rollouts=2))
        await publisher.start()
        for rollout_id in range(3):
            publisher.publish(
                ValueTrainingRollout(
                    samples=[ValueTrainingSample(token_ids=[1, 2], mask=[False, True], targets=[0.0, 1.0])],
                    rollout_id=rollout_id,
                    policy_version=0,
                    value_version=0,
                )
            )

        metrics = publisher.metrics()
        assert publisher.pending_rollouts == 2
        assert metrics["value/rollout_queue_enqueued"] == 3
        assert metrics["value/rollout_queue_sent"] == 0
        assert metrics["value/rollout_queue_dropped_oldest"] == 1
        assert metrics["value/rollout_queue_pending_tokens"] == 4

        await publisher.close()
        assert publisher.pending_rollouts == 0

    asyncio.run(run_test())


def test_connected_idle_trainer_leaves_the_bounded_producer_fifo_authoritative():
    async def run_test() -> None:
        config = _transport_config(max_pending_rollouts=3)
        receiver = ValueRolloutReceiver(config)
        publisher = ValueRolloutPublisher(config)
        await publisher.start()
        await asyncio.sleep(0.02)
        for rollout_id in range(10):
            publisher.publish(_rollout(rollout_id))
            await asyncio.sleep(0.001)

        assert publisher.sent == 0
        assert publisher.pending_rollouts == 3
        assert publisher.dropped_oldest == 7

        received = await _receive_response(receiver, 3)
        assert [rollout.rollout_id for rollout in received] == [7, 8, 9]
        assert publisher.sent == 3
        assert publisher.pending_rollouts == 0

        await publisher.close()
        receiver.close()

    asyncio.run(run_test())


def test_blocked_credited_reply_keeps_oldest_rollouts_droppable():
    async def run_test() -> None:
        publisher = ValueRolloutPublisher(ZMQValueTransportConfig(max_pending_rollouts=2))
        socket = _BlockedReplySocket(_RolloutRequest(request_id=0, limit=2))
        publisher._socket = socket  # type: ignore[assignment]
        publisher._responder_task = asyncio.create_task(publisher._respond_loop())
        publisher._responder_task.add_done_callback(publisher._on_responder_done)

        publisher.publish(_rollout(0))
        publisher.publish(_rollout(1))
        await asyncio.wait_for(socket.poll_started.wait(), timeout=1)
        publisher.publish(_rollout(2))

        assert publisher.pending_rollouts == 2
        assert publisher.dropped_oldest == 1

        socket.writable.set()
        await asyncio.wait_for(socket.sent.wait(), timeout=1)
        decoder = msgspec.msgpack.Decoder(type=ValueTrainingRollout)
        assert [decoder.decode(frame).rollout_id for frame in socket.frames[1:]] == [1, 2]

        await publisher.close()
        assert socket.closed

    asyncio.run(run_test())


def test_failed_immediate_reply_restores_only_rollouts_not_displaced_by_newer_arrivals():
    async def run_test() -> None:
        publisher = ValueRolloutPublisher(ZMQValueTransportConfig(max_pending_rollouts=2))
        socket = _AgainOnceReplySocket(_RolloutRequest(request_id=0, limit=2))
        publisher._socket = socket  # type: ignore[assignment]
        publisher._responder_task = asyncio.create_task(publisher._respond_loop())
        publisher._responder_task.add_done_callback(publisher._on_responder_done)

        publisher.publish(_rollout(0))
        publisher.publish(_rollout(1))
        await asyncio.wait_for(socket.first_send.wait(), timeout=1)
        publisher.publish(_rollout(2))
        publisher.publish(_rollout(3))
        assert socket.release_first_send is not None
        socket.release_first_send.set_exception(zmq.Again())
        await asyncio.wait_for(socket.sent.wait(), timeout=1)

        decoder = msgspec.msgpack.Decoder(type=ValueTrainingRollout)
        assert [decoder.decode(frame).rollout_id for frame in socket.frames[1:]] == [2, 3]
        assert publisher.dropped_oldest == 2
        assert publisher.sent == 2

        await publisher.close()

    asyncio.run(run_test())


def test_receiver_keeps_at_most_one_pull_request_outstanding():
    async def run_test() -> None:
        config = _transport_config()
        receiver = ValueRolloutReceiver(config)
        publisher = ValueRolloutPublisher(config)
        await publisher.start()
        await asyncio.sleep(0.02)

        assert receiver.receive_available(3) == []
        pending = receiver._pending_request
        assert pending is not None
        assert receiver.receive_available(1) == []
        assert receiver._pending_request is pending

        publisher.publish(_rollout(0))
        publisher.publish(_rollout(1))
        publisher.publish(_rollout(2))
        received = await _receive_response(receiver, 1)
        assert [rollout.rollout_id for rollout in received] == [0]
        assert [rollout.rollout_id for rollout in receiver.receive_available(1)] == [1]
        assert [rollout.rollout_id for rollout in receiver.receive_available(1)] == [2]
        assert publisher.pending_rollouts == 0

        await publisher.close()
        receiver.close()

    asyncio.run(run_test())


def test_receiver_rejects_malformed_correlated_responses():
    encode = msgspec.msgpack.encode
    rollout_frames = [encode(_rollout(0)), encode(_rollout(1))]
    cases = [
        ([encode(_RolloutResponse(request_id=1, count=1)), rollout_frames[0]], "does not match request"),
        ([encode(_RolloutResponse(request_id=0, count=2)), rollout_frames[0]], "declared 2 rollout"),
        ([encode(_RolloutResponse(request_id=0, count=2)), *rollout_frames], "exceeded request limit"),
    ]

    for frames, message in cases:
        receiver = ValueRolloutReceiver(_transport_config())
        receiver.socket.close(linger=0)
        receiver.socket = _ResponseSocket(frames)  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match=message):
                receiver.receive_available(1)
        finally:
            receiver.close()


def test_close_with_an_outstanding_real_pull_request_does_not_wait_for_a_response():
    async def run_test() -> None:
        config = _transport_config()
        receiver = ValueRolloutReceiver(config)
        publisher = ValueRolloutPublisher(config)
        await publisher.start()
        await asyncio.sleep(0.02)

        assert receiver.receive_available(1) == []
        assert receiver._pending_request is not None
        receiver.close()
        await asyncio.wait_for(publisher.close(), timeout=1)

    asyncio.run(run_test())


def test_publisher_close_contains_reported_responder_failure():
    class FailingPublisher(ValueRolloutPublisher):
        async def _respond_loop(self) -> None:
            raise RuntimeError("responder failed")

    async def run_test() -> None:
        publisher = FailingPublisher(ZMQValueTransportConfig())
        await publisher.start()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert publisher.metrics()["value/rollout_responder_failures"] == 1
        await publisher.close()

    asyncio.run(run_test())
