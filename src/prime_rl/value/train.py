from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import msgspec
import torch
import torch.distributed as dist

from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.trainer.model import DTYPE_MAP
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit
from prime_rl.value.trainer import ValueTrainerRuntime
from prime_rl.value.transport import LatestValueBatchReceiver
from prime_rl.value.types import ValueTrainingBatch


def _broadcast_batch(
    receiver: LatestValueBatchReceiver | None,
    decoder: msgspec.msgpack.Decoder,
) -> ValueTrainingBatch | None:
    world = get_world()
    payload: list[bytes | None] = [None]
    if world.is_master:
        assert receiver is not None
        batch = receiver.receive()
        payload[0] = msgspec.msgpack.encode(batch) if batch is not None else None
    dist.broadcast_object_list(payload, src=0, device=torch.device("cuda", world.local_rank))
    return decoder.decode(payload[0]) if payload[0] is not None else None


def _policy_run_finished() -> bool:
    path = os.environ.get("PRIME_RL_RUN_DONE_FILE")
    return path is not None and Path(path).exists()


@clean_exit
def train_value(config: ValueFunctionConfig) -> None:
    if config.model is None:
        raise ValueError("value_function.model must be resolved before starting the value trainer")
    trainer_placed = config.evaluator.placement == "trainer"
    run_done_file: Path | None = None
    if trainer_placed:
        raw_run_done_file = os.environ.get("PRIME_RL_RUN_DONE_FILE")
        if raw_run_done_file is None:
            raise ValueError("trainer-placed value evaluation must run through the managed RL launcher")
        run_done_file = Path(raw_run_done_file)

    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=trainer_placed or config.model.fsdp_cpu_offload,
    )
    torch.set_float32_matmul_precision(config.matmul_precision)
    resolve_ep(config.model)
    if trainer_placed:
        from prime_rl.value.colocated import validate_colocated_model

        validate_colocated_model(config.model)
    parallel_dims = get_parallel_dims(config.model)
    runtime = ValueTrainerRuntime(config, parallel_dims)

    if trainer_placed:
        from prime_rl.value.colocated import run_colocated

        assert run_done_file is not None
        run_colocated(config, runtime, run_done_file)
        return

    receiver = LatestValueBatchReceiver(config.transport) if world.is_master else None
    decoder = msgspec.msgpack.Decoder(type=ValueTrainingBatch)
    from prime_rl.value.weights import ValueWeightPublisher

    weight_publisher = ValueWeightPublisher(
        config.weight_broadcast,
        torch.device("cuda", world.local_rank),
        transfer_dtype=DTYPE_MAP[config.evaluator.dtype],
    )
    weight_publisher.publish(runtime.model, runtime.version)
    logger.info(f"Published initial value model version {runtime.version}")

    while not runtime.max_steps_reached:
        if _policy_run_finished():
            logger.info("Policy run completed; stopping value updates")
            break
        batch = _broadcast_batch(receiver, decoder)
        if batch is None:
            continue
        if not runtime.prepare_batch(batch):
            continue
        while runtime.can_step:
            runtime.step(weight_publisher)

    runtime.finish()
    if receiver is not None:
        receiver.close()


def main() -> None:
    """Torchrun entry point for the independent value trainer."""
    set_proc_title("ValueTrainer")
    train_value(cli(ValueFunctionConfig))


if __name__ == "__main__":
    main()
