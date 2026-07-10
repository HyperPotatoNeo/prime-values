from __future__ import annotations

import msgspec
import zmq

from prime_rl.configs.value import LatestZMQValueTransportConfig
from prime_rl.utils.logger import get_logger
from prime_rl.value.types import ValueTrainingBatch


class LatestValueBatchPublisher:
    """Non-blocking capacity-one trajectory publisher.

    ``ZMQ_CONFLATE`` keeps only the newest complete, single-frame message. A
    disconnected or saturated value trainer drops value work without applying
    backpressure to the orchestrator.
    """

    def __init__(self, config: LatestZMQValueTransportConfig):
        self.encoder = msgspec.msgpack.Encoder()
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.endpoint = f"tcp://{config.host}:{config.port}"
        self.socket.connect(self.endpoint)
        self.published = 0
        self.dropped = 0

    def publish(self, batch: ValueTrainingBatch) -> bool:
        try:
            self.socket.send(self.encoder.encode(batch), flags=zmq.DONTWAIT, copy=False)
        except zmq.Again:
            self.dropped += 1
            return False
        self.published += 1
        return True

    def close(self) -> None:
        self.socket.close(linger=0)


class LatestValueBatchReceiver:
    """Blocking receiver for the newest trajectory batch available."""

    def __init__(self, config: LatestZMQValueTransportConfig):
        self.decoder = msgspec.msgpack.Decoder(type=ValueTrainingBatch)
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, config.poll_timeout_ms)
        self.endpoint = f"tcp://{config.bind_host}:{config.port}"
        self.socket.bind(self.endpoint)
        get_logger().info(f"Value trainer listening for latest trajectories on {self.endpoint}")

    def receive(self) -> ValueTrainingBatch | None:
        try:
            payload = self.socket.recv(copy=False)
        except zmq.Again:
            return None
        return self.decoder.decode(payload)

    def close(self) -> None:
        self.socket.close(linger=0)
