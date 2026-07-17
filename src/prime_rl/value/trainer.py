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

if TYPE_CHECKING:
    from prime_rl.value.replay import ValueReplaySnapshot
    from prime_rl.value.weights import ValueWeightPublisher


@dataclass
class _PreparedBatch:
    batch: ValueTrainingBatch
    micro_batches: list[ValueMicroBatch]
    scale: int


@dataclass
class _ValueMetricState:
    loss_sum: torch.Tensor
    abs_error_sum: torch.Tensor
    squared_error_sum: torch.Tensor
    error_sum: torch.Tensor
    prediction_sum: torch.Tensor
    prediction_squared_sum: torch.Tensor
    target_sum: torch.Tensor
    target_squared_sum: torch.Tensor
    prediction_min: torch.Tensor
    prediction_max: torch.Tensor
    target_min: torch.Tensor
    target_max: torch.Tensor
    metric_count: torch.Tensor
    accuracy_sum: torch.Tensor
    accuracy_count: torch.Tensor
    entropy_sum: torch.Tensor
    confidence_sum: torch.Tensor
    classification_count: torch.Tensor

    @classmethod
    def empty(cls) -> _ValueMetricState:
        return cls(
            loss_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            abs_error_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            squared_error_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            error_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            prediction_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            prediction_squared_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            target_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            target_squared_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            prediction_min=torch.full((), torch.inf, dtype=torch.float32, device="cuda"),
            prediction_max=torch.full((), -torch.inf, dtype=torch.float32, device="cuda"),
            target_min=torch.full((), torch.inf, dtype=torch.float32, device="cuda"),
            target_max=torch.full((), -torch.inf, dtype=torch.float32, device="cuda"),
            metric_count=torch.zeros((), dtype=torch.float32, device="cuda"),
            accuracy_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            accuracy_count=torch.zeros((), dtype=torch.float32, device="cuda"),
            entropy_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            confidence_sum=torch.zeros((), dtype=torch.float32, device="cuda"),
            classification_count=torch.zeros((), dtype=torch.float32, device="cuda"),
        )

    def reduced(self, group: dist.ProcessGroup) -> _ValueMetricState:
        totals = torch.stack(
            [
                self.loss_sum,
                self.abs_error_sum,
                self.squared_error_sum,
                self.error_sum,
                self.prediction_sum,
                self.prediction_squared_sum,
                self.target_sum,
                self.target_squared_sum,
                self.metric_count,
                self.accuracy_sum,
                self.accuracy_count,
                self.entropy_sum,
                self.confidence_sum,
                self.classification_count,
            ]
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=group)
        mins = torch.stack([self.prediction_min, self.target_min])
        maxes = torch.stack([self.prediction_max, self.target_max])
        dist.all_reduce(mins, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(maxes, op=dist.ReduceOp.MAX, group=group)
        return _ValueMetricState(
            loss_sum=totals[0],
            abs_error_sum=totals[1],
            squared_error_sum=totals[2],
            error_sum=totals[3],
            prediction_sum=totals[4],
            prediction_squared_sum=totals[5],
            target_sum=totals[6],
            target_squared_sum=totals[7],
            metric_count=totals[8],
            accuracy_sum=totals[9],
            accuracy_count=totals[10],
            entropy_sum=totals[11],
            confidence_sum=totals[12],
            classification_count=totals[13],
            prediction_min=mins[0],
            prediction_max=maxes[0],
            target_min=mins[1],
            target_max=maxes[1],
        )


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
        self._last_checkpoint_version = resume_step
        self._active: _PreparedBatch | None = None
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
        return self._active is not None and not self.max_steps_reached

    def prepare_batch(self, batch: ValueTrainingBatch) -> bool:
        """Pack one optimizer batch on every rank."""
        if self._active is not None:
            raise RuntimeError("cannot replace a prepared value batch")
        assert self.config.model is not None
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
                get_logger().warning(
                    f"Skipping value replay batch spanning rollouts "
                    f"{batch.rollout_id_min}-{batch.rollout_id_max} with no trainable tokens"
                )
            return False

        self._active = _PreparedBatch(
            batch=batch,
            micro_batches=micro_batches,
            scale=scale,
        )
        return True

    def step(
        self,
        weight_publisher: ValueWeightPublisher | None = None,
        replay_snapshot: ValueReplaySnapshot | None = None,
    ) -> int:
        """Perform one complete value update, including optional publication and checkpointing."""
        if not self.can_step:
            raise RuntimeError("value trainer has no prepared update")
        assert self._active is not None
        assert self.config.model is not None
        active = self._active
        batch = active.batch
        update_started_at = time.perf_counter()
        source_value_lag_min = max(self.version - batch.value_version_max, 0)
        source_value_lag_max = max(self.version - batch.value_version_min, 0)
        metric_state = _ValueMetricState.empty()
        for micro_batch in active.micro_batches:
            input_ids, position_ids, mask, targets = _to_tensors(micro_batch)
            with maybe_activation_offloading(self.config.model.ac_offloading):
                logits = predict_value(self.model, input_ids, position_ids)
                logits = align_value_logits(logits, micro_batch.sequence_lengths)
                loss, metrics = compute_value_loss(logits, targets, mask, self.config.loss, active.scale)
            loss.backward()
            metric_state.loss_sum += loss.detach()
            metric_state.abs_error_sum += metrics["value/abs_error"].sum().to("cuda")
            metric_state.squared_error_sum += metrics["value/squared_error"].sum().to("cuda")
            metric_state.error_sum += metrics["value/error"].sum().to("cuda")
            predictions = metrics["value/prediction"].to("cuda")
            targets = metrics["value/target"].to("cuda")
            metric_state.prediction_sum += predictions.sum()
            metric_state.prediction_squared_sum += predictions.square().sum()
            metric_state.target_sum += targets.sum()
            metric_state.target_squared_sum += targets.square().sum()
            if predictions.numel() > 0:
                metric_state.prediction_min = torch.minimum(metric_state.prediction_min, predictions.min())
                metric_state.prediction_max = torch.maximum(metric_state.prediction_max, predictions.max())
                metric_state.target_min = torch.minimum(metric_state.target_min, targets.min())
                metric_state.target_max = torch.maximum(metric_state.target_max, targets.max())
            metric_state.metric_count += predictions.numel()
            if "value/accuracy" in metrics:
                metric_state.accuracy_sum += metrics["value/accuracy"].sum().to("cuda")
                metric_state.accuracy_count += metrics["value/accuracy"].numel()
                metric_state.entropy_sum += metrics["value/entropy"].sum().to("cuda")
                metric_state.confidence_sum += metrics["value/confidence"].sum().to("cuda")
                metric_state.classification_count += metrics["value/entropy"].numel()

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
        metric_state = metric_state.reduced(self._dp_group)
        if self.world.is_master:
            self._log_step(
                active,
                source_value_lag_min,
                source_value_lag_max,
                update_seconds,
                checkpoint_seconds,
                metric_state,
                grad_norm,
                zero_grad_ratio,
                replay_snapshot,
            )

        self._active = None
        return self.version

    def _save_checkpoint(self) -> None:
        assert self.ckpt_manager is not None
        self.ckpt_manager.save(self.version, self.model, [self.optimizer], self.scheduler, self.progress)
        self.ckpt_manager.mark_stable(self.version)
        self.ckpt_manager.maybe_clean()
        self._last_checkpoint_version = self.version

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
        self._save_checkpoint()
        return time.perf_counter() - started_at

    def _log_step(
        self,
        active: _PreparedBatch,
        source_value_lag_min: int,
        source_value_lag_max: int,
        update_seconds: float,
        checkpoint_seconds: float,
        metric_state: _ValueMetricState,
        grad_norm: torch.Tensor | None,
        zero_grad_ratio: float,
        replay_snapshot: ValueReplaySnapshot | None,
    ) -> None:
        batch = active.batch
        count = max(metric_state.metric_count.item(), 1.0)
        prediction_mean = metric_state.prediction_sum.item() / count
        target_mean = metric_state.target_sum.item() / count
        bias = metric_state.error_sum.item() / count
        mse = metric_state.squared_error_sum.item() / count
        prediction_variance = max(metric_state.prediction_squared_sum.item() / count - prediction_mean**2, 0.0)
        target_variance = max(metric_state.target_squared_sum.item() / count - target_mean**2, 0.0)
        error_variance = max(mse - bias**2, 0.0)
        payload = {
            "value/loss": metric_state.loss_sum.item(),
            "value/abs_error": metric_state.abs_error_sum.item() / count,
            "value/mae": metric_state.abs_error_sum.item() / count,
            "value/mse": mse,
            "value/rmse": mse**0.5,
            "value/bias": bias,
            "value/explained_variance": (1.0 - error_variance / target_variance if target_variance > 1e-12 else 0.0),
            "value/prediction_mean": prediction_mean,
            "value/prediction_std": prediction_variance**0.5,
            "value/prediction_min": metric_state.prediction_min.item(),
            "value/prediction_max": metric_state.prediction_max.item(),
            "value/target_mean": target_mean,
            "value/target_std": target_variance**0.5,
            "value/target_min": metric_state.target_min.item(),
            "value/target_max": metric_state.target_max.item(),
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
            "value/source_rollout_id_min": float(batch.rollout_id_min),
            "value/source_rollout_id_max": float(batch.rollout_id_max),
            "value/source_rollout_id_spread": float(batch.rollout_id_max - batch.rollout_id_min),
            "value/replay_attempt_min": float(batch.replay_attempt_min),
            "value/replay_attempt_max": float(batch.replay_attempt_max),
            "value/replay_attempt_mean": batch.replay_attempt_mean,
            "value/batch_tokens": float(active.scale),
            "value/batch_rollouts": float(batch.num_rollouts),
            "value/batch_samples": float(len(batch.samples)),
            "value/update_seconds": update_seconds,
            "value/tokens_per_second": active.scale / max(update_seconds, 1e-12),
            "value/total_tokens": float(self.progress.total_tokens),
            "value/total_samples": float(self.progress.total_samples),
            "optim/lr": self.optimizer.param_groups[0]["lr"],
            "optim/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
            "optim/zero_grad_ratio": zero_grad_ratio,
        }
        if replay_snapshot is not None:
            payload |= {
                "value/replay_size_rollouts": float(replay_snapshot.size),
                "value/replay_size_samples": float(replay_snapshot.samples),
                "value/replay_size_tokens": float(replay_snapshot.tokens),
                "value/replay_capacity_rollouts": float(replay_snapshot.capacity),
                "value/replay_refill_size_rollouts": float(replay_snapshot.refill_size),
                "value/replay_ready": float(replay_snapshot.ready),
                "value/replay_admitted_total": float(replay_snapshot.admitted),
                "value/replay_attempts_total": float(replay_snapshot.attempts),
                "value/replay_retired_total": float(replay_snapshot.retired),
                "value/replay_evicted_total": float(replay_snapshot.evicted),
            }
        if self.config.evaluator.placement == "trainer":
            payload["value/checkpoint_seconds"] = checkpoint_seconds
        if metric_state.accuracy_count.item() > 0:
            payload["value/accuracy"] = metric_state.accuracy_sum.item() / metric_state.accuracy_count.item()
        if metric_state.classification_count.item() > 0:
            payload["value/entropy"] = metric_state.entropy_sum.item() / metric_state.classification_count.item()
            payload["value/confidence"] = metric_state.confidence_sum.item() / metric_state.classification_count.item()
        self.monitor.log(payload, step=self.version)
        get_logger().info(
            f"Value version {self.version} | rollouts {batch.rollout_id_min}-{batch.rollout_id_max} "
            f"({batch.num_rollouts} sampled) | "
            f"loss {metric_state.loss_sum.item():.5f} | mae {payload['value/mae']:.5f} | "
            f"explained variance {payload['value/explained_variance']:.3f}"
        )

    def finish(self) -> None:
        """Collectively write a missing final checkpoint and close the monitor."""
        if self._finished:
            return
        if self.ckpt_manager is not None and self.version > 0 and self._last_checkpoint_version != self.version:
            get_logger().info(f"Writing final value checkpoint at version {self.version}")
            started_at = time.perf_counter()
            self._save_checkpoint()
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
