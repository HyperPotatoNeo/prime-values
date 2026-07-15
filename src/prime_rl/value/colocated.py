from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import msgspec
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule

from prime_rl.configs.trainer import ModelConfig
from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.trainer.model import value_model_supports_packing
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.value.batch import pack_value_inputs
from prime_rl.value.inference import predict_value_microbatches, reassemble_value_outputs
from prime_rl.value.service import ValueHTTPServer, ValueRequestBatch, ValueRequestService
from prime_rl.value.trainer import ValueTrainerRuntime
from prime_rl.value.transport import LatestValueBatchReceiver
from prime_rl.value.types import ValueEvaluationRequest, ValueTrainingBatch
from prime_rl.value.update_schedule import choose_next_operation

_IDLE_WAIT_SECONDS = 0.01
_MAX_IDLE_HEARTBEAT_SECONDS = 60.0
_ControlKind = Literal["train_new", "train_reuse", "infer", "idle", "stop"]


def validate_colocated_model(config: ModelConfig) -> None:
    """Reject trainer topologies not covered by the all-rank command loop."""
    if config.dp_replicate != 1:
        raise ValueError("trainer-placed value evaluation requires model.dp_replicate=1")
    if config.cp != 1:
        raise ValueError("trainer-placed value evaluation does not support context parallelism")
    if config.fsdp_cpu_offload:
        raise ValueError("trainer-placed value evaluation does not support FSDP CPU offload")
    if config.fp8:
        raise ValueError("trainer-placed value evaluation does not support FP8")
    if config.ep_comm_backend != "torch":
        raise ValueError("trainer-placed value evaluation does not support DeepEP")
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    text_config = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
    if any(
        getattr(candidate, field, None)
        for candidate in (model_config, text_config)
        for field in ("num_experts", "n_routed_experts", "num_local_experts", "num_experts_per_tok")
    ):
        raise ValueError("trainer-placed value evaluation does not support MoE models")
    if config.ep != 1:
        raise ValueError("trainer-placed value evaluation does not support expert parallelism")


class _ControlCommand(msgspec.Struct, array_like=True):
    command_id: int
    kind: _ControlKind
    payload: bytes | None = None


@dataclass
class _CoordinatorMetrics:
    inference_batches: int = 0
    inference_tokens: int = 0
    inference_seconds: float = 0.0
    training_seconds: float = 0.0
    idle_seconds: float = 0.0
    stale_training_batches: int = 0

    def snapshot(self, service: ValueRequestService, version: int) -> dict[str, float]:
        return service.metrics() | {
            "value/service_inference_batches": float(self.inference_batches),
            "value/service_inference_tokens": float(self.inference_tokens),
            "value/service_inference_seconds": self.inference_seconds,
            "value/service_training_seconds": self.training_seconds,
            "value/service_idle_seconds": self.idle_seconds,
            "value/service_stale_training_batches": float(self.stale_training_batches),
            "value/version": float(version),
        }


def _broadcast_command(command: _ControlCommand | None) -> _ControlCommand:
    world = get_world()
    payload = [msgspec.msgpack.encode(command) if world.is_master else None]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("value trainer received an empty control command")
    return msgspec.msgpack.decode(payload[0], type=_ControlCommand)


def _finalize_fsdp_inference(model: torch.nn.Module) -> None:
    if not isinstance(model, FSDPModule):
        raise TypeError("trainer-placed value inference requires a root FSDP model")
    # No-grad forwards record FSDP post-forward state but do not schedule the
    # post-backward callback that reshards and clears it.
    model._get_fsdp_state()._root_post_backward_final_callback()


def _run_inference(
    runtime: ValueTrainerRuntime,
    token_ids: list[list[int]],
) -> tuple[list[list[float]], int] | None:
    config = runtime.config
    assert config.model is not None
    world = get_world()
    data_mesh = runtime.parallel_dims.get_mesh("dp")
    if data_mesh.size() != world.world_size:
        raise RuntimeError("trainer-placed value inference requires the full trainer world")
    grid = pack_value_inputs(
        token_ids,
        seq_len=config.model.seq_len,
        world_size=world.world_size,
        pad_token_id=0,
        pack_sequences=value_model_supports_packing(runtime.model),
    )

    version = runtime.version
    was_training = runtime.model.training
    runtime.model.eval()
    try:
        with torch.no_grad():
            indexed = predict_value_microbatches(
                runtime.model,
                grid[data_mesh.get_local_rank()],
                device=torch.device("cuda", world.local_rank),
                loss=config.loss,
            )
        _finalize_fsdp_inference(runtime.model)
    finally:
        runtime.model.train(was_training)

    gathered: list[tuple[int, list[tuple[int, list[float]]]] | None] | None
    gathered = [None] * world.world_size if world.is_master else None
    dist.gather_object((version, indexed), gathered, dst=0)
    if not world.is_master:
        return None
    resolved = cast(list[tuple[int, list[tuple[int, list[float]]]]], gathered)
    versions = {rank_version for rank_version, _ in resolved}
    if versions != {version}:
        raise RuntimeError(f"value trainer ranks used inconsistent inference versions: {sorted(versions)}")
    all_indexed = [item for _, rank_values in resolved for item in rank_values]
    values = reassemble_value_outputs(all_indexed, [len(tokens) for tokens in token_ids])
    return values, version


def _model_vocab_size(runtime: ValueTrainerRuntime) -> int:
    model_config = runtime.model.config
    text_config = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
    return text_config.vocab_size


def run_colocated(
    config: ValueFunctionConfig,
    runtime: ValueTrainerRuntime,
    run_done_file: Path,
) -> None:
    """Alternate FIFO inference batches with updates on the live FSDP model."""
    assert config.model is not None
    world = get_world()
    service: ValueRequestService | None = None
    server: ValueHTTPServer | None = None
    server_thread: threading.Thread | None = None
    receiver: LatestValueBatchReceiver | None = None
    active_request: ValueRequestBatch | None = None
    stats = _CoordinatorMetrics()

    try:
        dist.barrier()
        if world.is_master:
            service = ValueRequestService(
                config.evaluator,
                seq_len=config.model.seq_len,
                vocab_size=_model_vocab_size(runtime),
                version=runtime.version,
            )
            port = config.evaluator.port
            server = ValueHTTPServer((config.evaluator.host, port), service)
            server_thread = threading.Thread(target=server.serve_forever, name="value-http", daemon=True)
            server_thread.start()
            if not runtime.max_steps_reached:
                receiver = LatestValueBatchReceiver(config.transport)
            get_logger().info(
                f"Serving value inference from trainer version {runtime.version} on "
                f"{config.evaluator.host}:{port}; dedicated serving settings are inactive"
            )

        pending_batch: ValueTrainingBatch | None = None
        last_batch_id: int | None = None
        last_operation: Literal["infer", "train"] | None = None
        serve_only = runtime.max_steps_reached
        next_command_id = 0
        idle_heartbeat_seconds = min(_MAX_IDLE_HEARTBEAT_SECONDS, config.dist_timeout_seconds / 2)
        last_command_at = time.monotonic()

        while True:
            command_kind: _ControlKind | None = None
            command_payload: bytes | None = None
            if world.is_master:
                assert service is not None
                if run_done_file.exists():
                    runtime.monitor.log(stats.snapshot(service, runtime.version), step=runtime.version)
                    service.close()
                    command_kind = "stop"
                else:
                    if runtime.max_steps_reached and not serve_only:
                        assert receiver is not None
                        receiver.close()
                        receiver = None
                        serve_only = True
                        get_logger().info(f"Value trainer reached max_steps; serving version {runtime.version}")

                    if not serve_only and not runtime.can_step and pending_batch is None:
                        assert receiver is not None
                        candidate = receiver.try_receive()
                        if candidate is not None:
                            if last_batch_id is not None and candidate.batch_id <= last_batch_id:
                                stats.stale_training_batches += 1
                            else:
                                pending_batch = candidate
                                last_batch_id = candidate.batch_id

                    operation = choose_next_operation(
                        has_inference=service.has_queued(),
                        has_training=runtime.can_step or pending_batch is not None,
                        last_operation=last_operation,
                    )
                    if operation == "infer":
                        active_request = service.take_batch(wait_for_first=False)
                        if active_request is not None:
                            request = ValueEvaluationRequest(token_ids=active_request.token_ids)
                            command_kind = "infer"
                            command_payload = msgspec.msgpack.encode(request)
                    elif operation == "train":
                        if runtime.can_step:
                            command_kind = "train_reuse"
                        else:
                            assert pending_batch is not None
                            command_kind = "train_new"
                            command_payload = msgspec.msgpack.encode(pending_batch)
                            pending_batch = None
                    else:
                        idle_started_at = time.perf_counter()
                        service.wait_for_work(_IDLE_WAIT_SECONDS)
                        stats.idle_seconds += time.perf_counter() - idle_started_at
                        if time.monotonic() - last_command_at < idle_heartbeat_seconds:
                            continue
                        command_kind = "idle"

                if command_kind is None:
                    command_kind = "idle"
            command = _ControlCommand(next_command_id, command_kind, command_payload) if world.is_master else None
            command = _broadcast_command(command)
            if command.command_id != next_command_id:
                raise RuntimeError(
                    f"value trainer expected control command {next_command_id}, got {command.command_id}"
                )
            next_command_id += 1
            last_command_at = time.monotonic()
            if command.kind == "stop":
                break
            if command.kind == "idle":
                continue
            if command.kind == "infer":
                if command.payload is None:
                    raise RuntimeError("inference command is missing its request")
                request = msgspec.msgpack.decode(command.payload, type=ValueEvaluationRequest)
                started_at = time.perf_counter()
                result = _run_inference(runtime, request.token_ids)
                if world.is_master:
                    assert service is not None and active_request is not None and result is not None
                    values, version = result
                    service.complete(active_request, values, version)
                    stats.inference_batches += 1
                    stats.inference_tokens += sum(map(len, request.token_ids))
                    stats.inference_seconds += time.perf_counter() - started_at
                    active_request = None
                last_operation = "infer"
                continue

            started_at = time.perf_counter()
            if command.kind == "train_new":
                if command.payload is None:
                    raise RuntimeError("new-batch command is missing its batch")
                batch = msgspec.msgpack.decode(command.payload, type=ValueTrainingBatch)
                runtime.prepare_batch(batch)
                if runtime.can_step:
                    runtime.step()
            elif command.kind == "train_reuse":
                if not runtime.can_step:
                    raise RuntimeError("value trainer received a reuse command without an active batch")
                runtime.step()
            if world.is_master:
                assert service is not None
                service.set_version(runtime.version)
                stats.training_seconds += time.perf_counter() - started_at
                runtime.monitor.log(stats.snapshot(service, runtime.version), step=runtime.version)
            last_operation = "train"

        runtime.finish()
    except Exception as error:
        if world.is_master and service is not None:
            if active_request is not None:
                service.fail(active_request, error)
            service.fail_service(error)
        raise
    finally:
        if world.is_master:
            if receiver is not None:
                receiver.close()
            if server is not None:
                server.shutdown()
                server.server_close()
            if server_thread is not None:
                server_thread.join()
