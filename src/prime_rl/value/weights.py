from __future__ import annotations

import socket
import struct
import time
from collections.abc import Iterator
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.value import NCCLValueWeightBroadcastConfig
from prime_rl.inference.vllm.worker.nccl import receive_integer, receive_state_dict
from prime_rl.trainer.conversion_utils import get_max_layer_num
from prime_rl.trainer.rl.broadcast.nccl import broadcast_integer, broadcast_state_dict, filter_state_dict_by_layers
from prime_rl.trainer.world import get_world
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable
from prime_rl.utils.vlm import get_layer_prefix

_VERSION = struct.Struct("!q")


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = connection.recv(remaining)
        except socket.timeout:
            # Timeout bounds each syscall without discarding a partially read
            # fixed-width frame or imposing an update-frequency deadline.
            continue
        if not chunk:
            raise ConnectionError("value weight control connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ValueWeightPublisher:
    """Layerwise raw-state NCCL fanout from FSDP value trainer to evaluators."""

    def __init__(
        self,
        config: NCCLValueWeightBroadcastConfig,
        device: torch.device,
        *,
        transfer_dtype: torch.dtype,
    ):
        self.world = get_world()
        self.transfer_dtype = transfer_dtype
        self.communicator: PyNcclCommunicator | None = None
        self.control_clients: list[socket.socket] = []
        if self.world.is_master:
            control_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            control_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            control_server.settimeout(config.timeout)
            control_server.bind(("0.0.0.0", config.control_port))
            control_server.listen(config.evaluator_world_size)
            for _ in range(config.evaluator_world_size):
                connection, _address = control_server.accept()
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(config.timeout)
                self.control_clients.append(connection)
            control_server.close()
            disable_nccl_p2p_if_unavailable()
            group = StatelessProcessGroup.create(
                host=config.host,
                port=config.port,
                rank=0,
                world_size=config.evaluator_world_size + 1,
                store_timeout=config.timeout,
            )
            self.communicator = PyNcclCommunicator(group, device=device)

    @torch.no_grad()
    def publish(self, model: nn.Module, version: int) -> None:
        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        if self.communicator is not None:
            announcement = _VERSION.pack(version)
            for connection in self.control_clients:
                connection.sendall(announcement)
            broadcast_integer(version, self.communicator)
            broadcast_integer(num_layers + 1, self.communicator)
        for _, layer in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            resolved: dict[str, Tensor] = {}
            for key, value in layer.items():
                full_value = cast(DTensor, value).full_tensor() if isinstance(value, DTensor) else value
                if full_value.is_floating_point():
                    full_value = full_value.to(self.transfer_dtype)
                resolved[key] = full_value
            if self.communicator is not None:
                broadcast_state_dict(resolved, self.communicator)


class ValueWeightReceiver:
    def __init__(
        self,
        config: NCCLValueWeightBroadcastConfig,
        *,
        evaluator_rank: int,
        device: torch.device,
    ):
        self.control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        deadline = time.monotonic() + config.timeout
        while True:
            try:
                self.control.connect((config.host, config.control_port))
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out connecting to value weight control at {config.host}:{config.control_port}"
                    )
                time.sleep(0.5)
        self.control.settimeout(config.timeout)
        disable_nccl_p2p_if_unavailable()
        group = StatelessProcessGroup.create(
            host=config.host,
            port=config.port,
            rank=evaluator_rank + 1,
            world_size=config.evaluator_world_size + 1,
            store_timeout=config.timeout,
        )
        self.communicator = PyNcclCommunicator(group, device=device)

    def receive(self) -> tuple[int, Iterator[dict[str, Tensor]]]:
        announced_version = _VERSION.unpack(_recv_exact(self.control, _VERSION.size))[0]
        version = receive_integer(self.communicator)
        if version != announced_version:
            raise RuntimeError(f"value weight version mismatch: control={announced_version}, nccl={version}")
        count = receive_integer(self.communicator)

        def layers() -> Iterator[dict[str, Tensor]]:
            for _ in range(count):
                yield dict(receive_state_dict(self.communicator))

        return version, layers()
