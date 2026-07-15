from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn
from torchtitan.distributed.utils import clip_grad_norm_

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.trainer.ckpt import CheckpointManager, setup_ckpt_managers
from prime_rl.trainer.model import predict_value, setup_value_model, value_model_supports_packing
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.runs import Progress
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import get_zero_gradient_ratio
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.utils import resolve_latest_ckpt_step
from prime_rl.value.batch import ValueMicroBatch, pack_value_samples
from prime_rl.value.math import align_value_logits, compute_value_loss, value_head_output_size
from prime_rl.value.types import ValueTrainingBatch
from prime_rl.value.update_schedule import updates_for_batch

if TYPE_CHECKING:
    from prime_rl.value.weights import ValueWeightPublisher


@dataclass
class _ActiveBatch:
    batch: ValueTrainingBatch
    micro_batches: list[ValueMicroBatch]
    scale: int
    source_batches_skipped: int
    updates: int
    reuse_step: int = 0


def _to_tensors(micro_batch: ValueMicroBatch) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda", torch.cuda.current_device())
    return (
        torch.tensor(micro_batch.input_ids, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(micro_batch.position_ids, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(micro_batch.mask, dtype=torch.bool, device=device).unsqueeze(0),
        torch.tensor(micro_batch.targets, dtype=torch.float32, device=device).unsqueeze(0),
    )


class ValueTrainerRuntime:
    """The live value model and one prepared optimizer batch."""

    def __init__(self, config: ValueFunctionConfig, parallel_dims: ParallelDims) -> None:
        if config.model is None:
            raise ValueError("value_function.model must be resolved before starting the value trainer")
        self.config = config
        self.parallel_dims = parallel_dims
        self.world = get_world()
        self.ckpt_manager: CheckpointManager | None = None
        resume_step: int | None = None
        if config.ckpt is not None:
            trainer_ckpt_config = CheckpointConfig(
                output_dir=config.output_dir,
                interval=config.ckpt.interval,
                weights=None,
                resume_step=config.ckpt.resume_step,
                keep_last=config.ckpt.keep_last,
            )
            self.ckpt_manager, _ = setup_ckpt_managers(config.output_dir, trainer_ckpt_config)
            if config.ckpt.resume_step == -1:
                resume_step = resolve_latest_ckpt_step(self.ckpt_manager.ckpt_dir)
            else:
                resume_step = config.ckpt.resume_step

        get_logger().info(f"Initializing value model {config.model.name} in {self.world}")
        self.model: nn.Module = setup_value_model(
            config.model,
            parallel_dims,
            output_size=value_head_output_size(config.loss),
            loading_from_checkpoint_later=resume_step is not None,
        )
        self.optimizer = setup_optimizer(
            config.optim,
            list(self.model.named_parameters()),
            parallel_dims,
            cpu_offload=config.model.optim_cpu_offload,
        )
        self.scheduler = setup_scheduler(self.optimizer, config.scheduler, config.max_steps, config.optim.lr)
        self.monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)
        self.progress = Progress(step=0)
        if resume_step is not None:
            assert self.ckpt_manager is not None
            self.ckpt_manager.load(resume_step, self.model, [self.optimizer], self.scheduler, self.progress)
            self.progress.step = resume_step
            get_logger().info(f"Resumed value trainer from version {resume_step}")

        data_mesh = parallel_dims.get_mesh("dp")
        self._data_world_size = data_mesh.size()
        self._data_rank = data_mesh.get_local_rank()
        self._dp_group = parallel_dims.get_mesh("dp_cp").get_group()
        self._last_batch_id: int | None = None
        self._last_checkpoint_version = resume_step
        self._active: _ActiveBatch | None = None
        self._finished = False
        self.optimizer.zero_grad()

    @property
    def version(self) -> int:
        return self.progress.step

    @property
    def max_steps_reached(self) -> bool:
        return self.config.max_steps is not None and self.version >= self.config.max_steps

    @property
    def can_step(self) -> bool:
        return (
            self._active is not None and self._active.reuse_step < self._active.updates and not self.max_steps_reached
        )

    def prepare_batch(self, batch: ValueTrainingBatch) -> bool:
        """Pack a new batch on every rank and retain it for its configured updates."""
        if self.can_step:
            raise RuntimeError("cannot replace a value batch with pending reuse updates")
        assert self.config.model is not None
        source_batches_skipped = 0 if self._last_batch_id is None else max(batch.batch_id - self._last_batch_id - 1, 0)
        self._last_batch_id = batch.batch_id
        grid = pack_value_samples(
            batch.samples,
            seq_len=self.config.model.seq_len,
            world_size=self._data_world_size,
            pad_token_id=0,
            pack_sequences=value_model_supports_packing(self.model),
        )
        micro_batches = grid[self._data_rank]
        local_scale = sum(sum(micro_batch.mask) for micro_batch in micro_batches)
        global_scale = torch.tensor(local_scale, dtype=torch.int64, device="cuda")
        dist.all_reduce(global_scale, op=dist.ReduceOp.SUM, group=self._dp_group)
        scale = int(global_scale.item())
        if scale == 0:
            self._active = None
            if self.world.is_master:
                get_logger().warning(f"Skipping value batch {batch.batch_id} with no trainable tokens")
            return False

        self._active = _ActiveBatch(
            batch=batch,
            micro_batches=micro_batches,
            scale=scale,
            source_batches_skipped=source_batches_skipped,
            updates=updates_for_batch(
                value_version=self.version,
                warmup_updates=self.config.warmup_updates,
                updates_per_batch=self.config.updates_per_batch,
            ),
        )
        return True

    def step(self, weight_publisher: ValueWeightPublisher | None = None) -> int:
        """Perform one complete value update, including optional publication and checkpointing."""
        if not self.can_step:
            raise RuntimeError("value trainer has no prepared update")
        assert self._active is not None
        assert self.config.model is not None
        active = self._active
        batch = active.batch
        reuse_step = active.reuse_step
        update_started_at = time.perf_counter()
        source_value_lag_min = max(self.version - batch.value_version_max, 0)
        source_value_lag_max = max(self.version - batch.value_version_min, 0)
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
        for micro_batch in active.micro_batches:
            input_ids, position_ids, mask, targets = _to_tensors(micro_batch)
            with maybe_activation_offloading(self.config.model.ac_offloading):
                logits = predict_value(self.model, input_ids, position_ids)
                logits = align_value_logits(logits, micro_batch.sequence_lengths)
                loss, metrics = compute_value_loss(logits, targets, mask, self.config.loss, active.scale)
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

        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(self.parallel_dims.fsdp_gradient_divide_factor)

        grad_norm = None
        if self.config.optim.max_norm is not None:
            grad_norm = clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.optim.max_norm,
                ep_enabled=self.parallel_dims.ep_enabled,
            )
            if grad_norm.device.type == "cpu":
                grad_norm = grad_norm.cuda()
        zero_grad_ratio = get_zero_gradient_ratio(self.model.parameters(), self.parallel_dims.dp_replicate)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        self.progress.step += 1
        self.progress.total_tokens += active.scale
        self.progress.total_samples += len(batch.samples)
        if weight_publisher is not None:
            weight_publisher.publish(self.model, self.version)
        checkpoint_seconds = self._checkpoint_if_due()

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
        dist.all_reduce(metric_totals, op=dist.ReduceOp.SUM, group=self._dp_group)
        metric_mins = torch.stack([prediction_min, target_min])
        metric_maxes = torch.stack([prediction_max, target_max])
        dist.all_reduce(metric_mins, op=dist.ReduceOp.MIN, group=self._dp_group)
        dist.all_reduce(metric_maxes, op=dist.ReduceOp.MAX, group=self._dp_group)
        if self.world.is_master:
            self._log_step(
                active,
                reuse_step,
                source_value_lag_min,
                source_value_lag_max,
                update_seconds,
                checkpoint_seconds,
                metric_totals,
                metric_mins,
                metric_maxes,
                grad_norm,
                zero_grad_ratio,
            )

        active.reuse_step += 1
        if active.reuse_step == active.updates or self.max_steps_reached:
            self._active = None
        return self.version

    def _checkpoint_if_due(self) -> float:
        if self.ckpt_manager is None:
            return 0.0
        interval_due = (
            self.config.ckpt is not None
            and self.config.ckpt.interval is not None
            and self.version % self.config.ckpt.interval == 0
        )
        if not interval_due and not self.max_steps_reached:
            return 0.0
        started_at = time.perf_counter()
        self.ckpt_manager.save(self.version, self.model, [self.optimizer], self.scheduler, self.progress)
        self.ckpt_manager.mark_stable(self.version)
        self.ckpt_manager.maybe_clean()
        self._last_checkpoint_version = self.version
        return time.perf_counter() - started_at

    def _log_step(
        self,
        active: _ActiveBatch,
        reuse_step: int,
        source_value_lag_min: int,
        source_value_lag_max: int,
        update_seconds: float,
        checkpoint_seconds: float,
        metric_totals: torch.Tensor,
        metric_mins: torch.Tensor,
        metric_maxes: torch.Tensor,
        grad_norm: torch.Tensor | None,
        zero_grad_ratio: float,
    ) -> None:
        batch = active.batch
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
            "value/explained_variance": (1.0 - error_variance / target_variance if target_variance > 1e-12 else 0.0),
            "value/prediction_mean": prediction_mean,
            "value/prediction_std": prediction_variance**0.5,
            "value/prediction_min": metric_mins[0].item(),
            "value/prediction_max": metric_maxes[0].item(),
            "value/target_mean": target_mean,
            "value/target_std": target_variance**0.5,
            "value/target_min": metric_mins[1].item(),
            "value/target_max": metric_maxes[1].item(),
            "value/version": float(self.version),
            "value/source_policy_version": float(batch.policy_version_max),
            "value/source_policy_version_min": float(batch.policy_version_min),
            "value/source_policy_version_max": float(batch.policy_version_max),
            "value/source_policy_version_spread": float(batch.policy_version_max - batch.policy_version_min),
            "value/source_value_version": float(batch.value_version_max),
            "value/source_value_version_min": float(batch.value_version_min),
            "value/source_value_version_max": float(batch.value_version_max),
            "value/source_value_version_spread": float(batch.value_version_max - batch.value_version_min),
            "value/source_value_lag": float(source_value_lag_max),
            "value/source_value_lag_min": float(source_value_lag_min),
            "value/source_value_lag_max": float(source_value_lag_max),
            "value/batch_id": float(batch.batch_id),
            "value/batch_tokens": float(active.scale),
            "value/batch_rollouts": float(batch.num_rollouts),
            "value/batch_samples": float(len(batch.samples)),
            "value/source_batches_skipped": float(active.source_batches_skipped),
            "value/reuse_step": float(reuse_step),
            "value/update_seconds": update_seconds,
            "value/tokens_per_second": active.scale / max(update_seconds, 1e-12),
            "value/total_tokens": float(self.progress.total_tokens),
            "value/total_samples": float(self.progress.total_samples),
            "optim/lr": self.optimizer.param_groups[0]["lr"],
            "optim/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
            "optim/zero_grad_ratio": zero_grad_ratio,
        }
        if self.config.evaluator.placement == "trainer":
            payload["value/checkpoint_seconds"] = checkpoint_seconds
        if metric_totals[10].item() > 0:
            payload["value/accuracy"] = metric_totals[9].item() / metric_totals[10].item()
        if metric_totals[13].item() > 0:
            payload["value/entropy"] = metric_totals[11].item() / metric_totals[13].item()
            payload["value/confidence"] = metric_totals[12].item() / metric_totals[13].item()
        self.monitor.log(payload, step=self.version)
        get_logger().info(
            f"Value version {self.version} | batch {batch.batch_id} | "
            f"rollouts {batch.num_rollouts} | reuse {reuse_step + 1}/{active.updates} | "
            f"loss {metric_totals[0].item():.5f} | mae {payload['value/mae']:.5f} | "
            f"explained variance {payload['value/explained_variance']:.3f}"
        )

    def finish(self) -> None:
        """Collectively write a missing final checkpoint and close the monitor."""
        if self._finished:
            return
        if self.ckpt_manager is not None and self.version > 0 and self._last_checkpoint_version != self.version:
            get_logger().info(f"Writing final value checkpoint at version {self.version}")
            started_at = time.perf_counter()
            self.ckpt_manager.save(self.version, self.model, [self.optimizer], self.scheduler, self.progress)
            self.ckpt_manager.mark_stable(self.version)
            self.ckpt_manager.maybe_clean()
            self._last_checkpoint_version = self.version
            if self.world.is_master and self.config.evaluator.placement == "trainer":
                self.monitor.log(
                    {
                        "value/checkpoint_seconds": time.perf_counter() - started_at,
                        "value/version": float(self.version),
                    },
                    step=self.version,
                )
        self.monitor.close()
        self._finished = True
