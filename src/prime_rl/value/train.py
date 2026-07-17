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
from prime_rl.value.coordinator import admit_available_rollouts
from prime_rl.value.replay import ValueReplayBuffer, ValueReplaySnapshot
from prime_rl.value.trainer import ValueTrainerRuntime
from prime_rl.value.transport import ValueRolloutReceiver
from prime_rl.value.types import ValueTrainingBatch


def _broadcast_batch(
    receiver: ValueRolloutReceiver | None,
    replay: ValueReplayBuffer | None,
    decoder: msgspec.msgpack.Decoder,
) -> tuple[ValueTrainingBatch | None, ValueReplaySnapshot | None, bool]:
    world = get_world()
    control: list[tuple[bool, bytes | None] | None] = [None]
    snapshot: ValueReplaySnapshot | None = None
    if world.is_master:
        assert receiver is not None and replay is not None
        stop = _policy_run_finished()
        encoded_batch: bytes | None = None
        if not stop:
            filling = not replay.can_sample
            admit_available_rollouts(receiver, replay, wait_for_first=filling)
            if replay.can_sample:
                batch = replay.sample()
                snapshot = replay.snapshot()
                encoded_batch = msgspec.msgpack.encode(batch)
        control[0] = (stop, encoded_batch)
    dist.broadcast_object_list(control, src=0, device=torch.device("cuda", world.local_rank))
    if control[0] is None:
        raise RuntimeError("value trainer received an empty coordinator control")
    stop, encoded_batch = control[0]
    batch = decoder.decode(encoded_batch) if encoded_batch is not None else None
    return batch, snapshot, stop


def _policy_run_finished() -> bool:
    path = os.environ.get("PRIME_RL_RUN_DONE_FILE")
    return path is not None and Path(path).exists()


@clean_exit
def train_value(config: ValueFunctionConfig) -> None:
    if config.model is None:
        raise ValueError("value_function.model must be resolved before starting the value trainer")
    batch_size = config.batch_size
    if batch_size is None:
        raise ValueError("value_function.batch_size must be resolved before starting the value trainer")
    replay_capacity = config.replay.resolved_capacity
    replay_refill_size = config.replay.resolved_refill_size
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

    receiver = ValueRolloutReceiver(config.transport) if world.is_master else None
    replay: ValueReplayBuffer | None = None
    if world.is_master:
        replay = ValueReplayBuffer(
            batch_size=batch_size,
            capacity=replay_capacity,
            refill_size=replay_refill_size,
            max_updates_per_rollout=config.replay.max_updates_per_rollout,
            seed=config.replay.seed,
        )
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
        batch, replay_snapshot, stop = _broadcast_batch(receiver, replay, decoder)
        if stop:
            if world.is_master:
                logger.info("Policy run completed; stopping value updates")
            break
        if batch is None:
            continue
        if not runtime.prepare_batch(batch):
            continue
        runtime.step(weight_publisher, replay_snapshot)

    runtime.finish()
    if receiver is not None:
        receiver.close()


def main() -> None:
    """Torchrun entry point for the independent value trainer."""
    set_proc_title("ValueTrainer")
    train_value(cli(ValueFunctionConfig))


if __name__ == "__main__":
    main()
