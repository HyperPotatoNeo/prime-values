from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path

import msgspec
import torch
import torch.distributed as dist
from torchtitan.distributed.utils import clip_grad_norm_

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.trainer.ckpt import CheckpointManager, setup_ckpt_managers
from prime_rl.trainer.model import DTYPE_MAP, predict_value, setup_value_model, value_model_supports_packing
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.runs import Progress
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import get_zero_gradient_ratio, setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit, resolve_latest_ckpt_step
from prime_rl.value.batch import ValueMicroBatch, pack_value_samples
from prime_rl.value.math import align_value_logits, compute_value_loss, value_head_output_size
from prime_rl.value.transport import LatestValueBatchReceiver
from prime_rl.value.types import ValueTrainingBatch
from prime_rl.value.weights import ValueWeightPublisher


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


def _to_tensors(micro_batch: ValueMicroBatch) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda", torch.cuda.current_device())
    return (
        torch.tensor(micro_batch.input_ids, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(micro_batch.position_ids, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(micro_batch.mask, dtype=torch.bool, device=device).unsqueeze(0),
        torch.tensor(micro_batch.targets, dtype=torch.float32, device=device).unsqueeze(0),
    )


@clean_exit
def train_value(config: ValueFunctionConfig) -> None:
    if config.model is None:
        raise ValueError("value_function.model must be resolved before starting the value trainer")
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.fsdp_cpu_offload,
    )
    torch.set_float32_matmul_precision(config.matmul_precision)
    resolve_ep(config.model)
    parallel_dims = get_parallel_dims(config.model)

    ckpt_manager: CheckpointManager | None = None
    resume_step: int | None = None
    if config.ckpt is not None:
        trainer_ckpt_config = CheckpointConfig(
            output_dir=config.output_dir,
            interval=config.ckpt.interval,
            weights=None,
            resume_step=config.ckpt.resume_step,
            keep_last=config.ckpt.keep_last,
        )
        ckpt_manager, _ = setup_ckpt_managers(config.output_dir, trainer_ckpt_config)
        if config.ckpt.resume_step == -1:
            assert ckpt_manager is not None
            resume_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
        else:
            resume_step = config.ckpt.resume_step

    logger.info(f"Initializing value model {config.model.name} in {world}")
    model = setup_value_model(
        config.model,
        parallel_dims,
        output_size=value_head_output_size(config.loss),
        loading_from_checkpoint_later=resume_step is not None,
    )
    optimizer = setup_optimizer(
        config.optim,
        list(model.named_parameters()),
        parallel_dims,
        cpu_offload=config.model.optim_cpu_offload,
    )
    scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)
    progress = Progress(step=0)
    if resume_step is not None:
        assert ckpt_manager is not None
        ckpt_manager.load(resume_step, model, [optimizer], scheduler, progress)
        progress.step = resume_step
        logger.info(f"Resumed value trainer from version {resume_step}")

    receiver = LatestValueBatchReceiver(config.transport) if world.is_master else None
    decoder = msgspec.msgpack.Decoder(type=ValueTrainingBatch)
    weight_publisher = ValueWeightPublisher(
        config.weight_broadcast,
        torch.device("cuda", world.local_rank),
        transfer_dtype=DTYPE_MAP[config.evaluator.dtype],
    )
    value_version = progress.step
    weight_publisher.publish(model, value_version)
    logger.info(f"Published initial value model version {value_version}")

    data_mesh = parallel_dims.get_mesh("dp")
    data_world_size = data_mesh.size()
    data_rank = data_mesh.get_local_rank()
    dp_group = parallel_dims.get_mesh("dp_cp").get_group()
    optimizer.zero_grad()
    last_batch_id: int | None = None
    last_checkpoint_version = resume_step

    while config.max_steps is None or value_version < config.max_steps:
        if _policy_run_finished():
            logger.info("Policy run completed; stopping value updates")
            break
        batch = _broadcast_batch(receiver, decoder)
        if batch is None:
            continue
        source_batches_skipped = 0 if last_batch_id is None else max(batch.batch_id - last_batch_id - 1, 0)
        last_batch_id = batch.batch_id
        grid = pack_value_samples(
            batch.samples,
            seq_len=config.model.seq_len,
            world_size=data_world_size,
            pad_token_id=0,
            pack_sequences=value_model_supports_packing(model),
        )
        micro_batches = grid[data_rank]

        local_scale = sum(sum(micro_batch.mask) for micro_batch in micro_batches)
        global_scale = torch.tensor(local_scale, dtype=torch.int64, device="cuda")
        dist.all_reduce(global_scale, op=dist.ReduceOp.SUM, group=dp_group)
        scale = int(global_scale.item())
        if scale == 0:
            if world.is_master:
                logger.warning(f"Skipping value batch {batch.batch_id} with no trainable tokens")
            continue

        for reuse_step in range(config.updates_per_batch):
            if config.max_steps is not None and value_version >= config.max_steps:
                break
            update_started_at = time.perf_counter()
            source_value_lag = max(value_version - batch.value_version, 0)
            loss_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            abs_error_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            squared_error_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            error_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            prediction_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            prediction_squared_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            target_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            target_squared_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            prediction_min = torch.full((), torch.inf, dtype=torch.float32, device="cuda")
            prediction_max = torch.full((), -torch.inf, dtype=torch.float32, device="cuda")
            target_min = torch.full((), torch.inf, dtype=torch.float32, device="cuda")
            target_max = torch.full((), -torch.inf, dtype=torch.float32, device="cuda")
            metric_count = torch.zeros((), dtype=torch.float32, device="cuda")
            accuracy_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            accuracy_count = torch.zeros((), dtype=torch.float32, device="cuda")
            entropy_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            confidence_sum = torch.zeros((), dtype=torch.float32, device="cuda")
            classification_count = torch.zeros((), dtype=torch.float32, device="cuda")
            for micro_batch in micro_batches:
                input_ids, position_ids, mask, targets = _to_tensors(micro_batch)
                with maybe_activation_offloading(config.model.ac_offloading):
                    logits = predict_value(model, input_ids, position_ids)
                    logits = align_value_logits(logits, micro_batch.sequence_lengths)
                    loss, metrics = compute_value_loss(logits, targets, mask, config.loss, scale)
                loss.backward()
                loss_sum += loss.detach()
                abs_error_sum += metrics["value/abs_error"].sum().to("cuda")
                squared_error_sum += metrics["value/squared_error"].sum().to("cuda")
                error_sum += metrics["value/error"].sum().to("cuda")
                predictions = metrics["value/prediction"].to("cuda")
                targets = metrics["value/target"].to("cuda")
                prediction_sum += predictions.sum()
                prediction_squared_sum += predictions.square().sum()
                target_sum += targets.sum()
                target_squared_sum += targets.square().sum()
                if predictions.numel() > 0:
                    prediction_min = torch.minimum(prediction_min, predictions.min())
                    prediction_max = torch.maximum(prediction_max, predictions.max())
                    target_min = torch.minimum(target_min, targets.min())
                    target_max = torch.maximum(target_max, targets.max())
                metric_count += predictions.numel()
                if "value/accuracy" in metrics:
                    accuracy_sum += metrics["value/accuracy"].sum().to("cuda")
                    accuracy_count += metrics["value/accuracy"].numel()
                    entropy_sum += metrics["value/entropy"].sum().to("cuda")
                    confidence_sum += metrics["value/confidence"].sum().to("cuda")
                    classification_count += metrics["value/entropy"].numel()

            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(parallel_dims.fsdp_gradient_divide_factor)

            grad_norm = None
            if config.optim.max_norm is not None:
                grad_norm = clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.optim.max_norm,
                    ep_enabled=parallel_dims.ep_enabled,
                )
                if grad_norm.device.type == "cpu":
                    grad_norm = grad_norm.cuda()
            zero_grad_ratio = get_zero_gradient_ratio(model.parameters(), parallel_dims.dp_replicate)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            value_version += 1
            progress.step = value_version
            progress.total_tokens += int(global_scale.item())
            progress.total_samples += len(batch.samples)
            weight_publisher.publish(model, value_version)
            if ckpt_manager is not None and (
                (
                    config.ckpt is not None
                    and config.ckpt.interval is not None
                    and value_version % config.ckpt.interval == 0
                )
                or (config.max_steps is not None and value_version == config.max_steps)
            ):
                ckpt_manager.save(value_version, model, [optimizer], scheduler, progress)
                ckpt_manager.mark_stable(value_version)
                ckpt_manager.maybe_clean()
                last_checkpoint_version = value_version
            update_seconds = time.perf_counter() - update_started_at
            metric_totals = torch.stack(
                [
                    loss_sum,
                    abs_error_sum,
                    squared_error_sum,
                    error_sum,
                    prediction_sum,
                    prediction_squared_sum,
                    target_sum,
                    target_squared_sum,
                    metric_count,
                    accuracy_sum,
                    accuracy_count,
                    entropy_sum,
                    confidence_sum,
                    classification_count,
                ]
            )
            dist.all_reduce(metric_totals, op=dist.ReduceOp.SUM, group=dp_group)
            metric_mins = torch.stack([prediction_min, target_min])
            metric_maxes = torch.stack([prediction_max, target_max])
            dist.all_reduce(metric_mins, op=dist.ReduceOp.MIN, group=dp_group)
            dist.all_reduce(metric_maxes, op=dist.ReduceOp.MAX, group=dp_group)
            if world.is_master:
                count = max(metric_totals[8].item(), 1.0)
                prediction_mean = metric_totals[4].item() / count
                target_mean = metric_totals[6].item() / count
                bias = metric_totals[3].item() / count
                mse = metric_totals[2].item() / count
                prediction_variance = max(metric_totals[5].item() / count - prediction_mean**2, 0.0)
                target_variance = max(metric_totals[7].item() / count - target_mean**2, 0.0)
                error_variance = max(mse - bias**2, 0.0)
                payload = {
                    "value/loss": metric_totals[0].item(),
                    "value/abs_error": metric_totals[1].item() / count,
                    "value/mae": metric_totals[1].item() / count,
                    "value/mse": mse,
                    "value/rmse": mse**0.5,
                    "value/bias": bias,
                    "value/explained_variance": (
                        1.0 - error_variance / target_variance if target_variance > 1e-12 else 0.0
                    ),
                    "value/prediction_mean": prediction_mean,
                    "value/prediction_std": prediction_variance**0.5,
                    "value/prediction_min": metric_mins[0].item(),
                    "value/prediction_max": metric_maxes[0].item(),
                    "value/target_mean": target_mean,
                    "value/target_std": target_variance**0.5,
                    "value/target_min": metric_mins[1].item(),
                    "value/target_max": metric_maxes[1].item(),
                    "value/version": float(value_version),
                    "value/source_policy_version": float(batch.policy_version),
                    "value/source_value_version": float(batch.value_version),
                    "value/source_value_lag": float(source_value_lag),
                    "value/batch_id": float(batch.batch_id),
                    "value/batch_tokens": float(scale),
                    "value/batch_samples": float(len(batch.samples)),
                    "value/source_batches_skipped": float(source_batches_skipped),
                    "value/reuse_step": float(reuse_step),
                    "value/update_seconds": update_seconds,
                    "value/tokens_per_second": scale / max(update_seconds, 1e-12),
                    "value/total_tokens": float(progress.total_tokens),
                    "value/total_samples": float(progress.total_samples),
                    "optim/lr": optimizer.param_groups[0]["lr"],
                    "optim/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
                    "optim/zero_grad_ratio": zero_grad_ratio,
                }
                if metric_totals[10].item() > 0:
                    payload["value/accuracy"] = metric_totals[9].item() / metric_totals[10].item()
                if metric_totals[13].item() > 0:
                    payload["value/entropy"] = metric_totals[11].item() / metric_totals[13].item()
                    payload["value/confidence"] = metric_totals[12].item() / metric_totals[13].item()
                monitor.log(payload, step=value_version)
                logger.info(
                    f"Value version {value_version} | batch {batch.batch_id} | "
                    f"reuse {reuse_step + 1}/{config.updates_per_batch} | "
                    f"loss {metric_totals[0].item():.5f} | mae {payload['value/mae']:.5f} | "
                    f"explained variance {payload['value/explained_variance']:.3f}"
                )

    if ckpt_manager is not None and value_version > 0 and last_checkpoint_version != value_version:
        logger.info(f"Writing final value checkpoint at version {value_version}")
        ckpt_manager.save(value_version, model, [optimizer], scheduler, progress)
        ckpt_manager.mark_stable(value_version)
        ckpt_manager.maybe_clean()

    if receiver is not None:
        receiver.close()
    monitor.close()


def main() -> None:
    """Torchrun entry point for the independent value trainer."""
    set_proc_title("ValueTrainer")
    train_value(cli(ValueFunctionConfig))


if __name__ == "__main__":
    main()
