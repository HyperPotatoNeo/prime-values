import math
from pathlib import Path
from typing import Annotated, Literal, TypeAlias
from urllib.parse import urlparse

from pydantic import Field, model_validator

from prime_rl.configs.shared import EnvVars, TrainerLogConfig, WandbConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from prime_rl.utils.config import BaseConfig


class MSEValueLossConfig(BaseConfig):
    type: Literal["mse"] = "mse"
    """Regress scalar lambda-return targets with mean-squared error."""


class ClassificationValueLossConfig(BaseConfig):
    type: Literal["classification"] = "classification"
    """Predict a categorical return distribution and use its expectation as value."""

    reward_range: tuple[float, float] = (0.0, 1.0)
    """Closed range represented by the categorical support."""

    num_bins: int = Field(2, ge=2)
    """Number of uniformly spaced support points, including both range endpoints."""

    @model_validator(mode="after")
    def validate_reward_range(self):
        low, high = self.reward_range
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("value_function.loss.reward_range must be increasing")
        return self


ValueLossConfig: TypeAlias = Annotated[
    MSEValueLossConfig | ClassificationValueLossConfig,
    Field(discriminator="type"),
]


class LatestZMQValueTransportConfig(BaseConfig):
    type: Literal["zmq_latest"] = "zmq_latest"

    host: str = "127.0.0.1"
    """Value-trainer host as reached by the orchestrator."""

    bind_host: str = "0.0.0.0"
    """Interface on which the value trainer binds."""

    port: int = Field(29610, ge=1, le=65535)
    """Dedicated latest-only full-batch trajectory port."""

    poll_timeout_ms: int = Field(1000, ge=1)
    """Receiver poll interval, allowing graceful policy-run shutdown checks."""


class NCCLValueWeightBroadcastConfig(BaseConfig):
    type: Literal["nccl"] = "nccl"

    host: str = "127.0.0.1"
    """Value-trainer rendezvous host as reached by evaluator replicas."""

    port: int = Field(29611, ge=1, le=65535)

    control_port: int = Field(29613, ge=1, le=65535)
    """CPU-side notification channel; prevents idle NCCL receives from occupying evaluator GPUs."""

    timeout: int = Field(1200, ge=1)

    evaluator_world_size: int = Field(1, ge=1)
    """Number of dedicated evaluator replicas receiving each update; trainer placement requires one endpoint."""


class ValueEvaluatorConfig(BaseConfig):
    placement: Literal["dedicated", "trainer"] = "dedicated"
    """Run a dedicated serving copy or serve between updates on the value trainer."""

    base_url: list[str] = ["http://127.0.0.1:29612"]
    """Evaluator HTTP endpoints used round-robin by the orchestrator."""

    host: str = "0.0.0.0"
    """HTTP bind interface for a locally launched evaluator."""

    port: int = Field(29612, ge=1, le=65535)

    max_batch_tokens: int = Field(32768, ge=1)
    """FIFO coalescing ceiling. A larger single request is served alone."""

    batch_wait_ms: float = Field(2.0, ge=0)
    """Small collection window used to merge concurrent rollout requests."""

    max_concurrency: int = Field(64, ge=1)
    """Maximum in-flight HTTP requests issued by the orchestrator."""

    max_pending_requests: int = Field(64, ge=1)
    """Maximum queued plus running requests accepted by one evaluator endpoint."""

    max_pending_tokens: int = Field(1_048_576, ge=1)
    """Maximum unpadded tokens across queued plus running requests."""

    request_timeout: float = Field(600.0, gt=0)

    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    """Dedicated serving-copy dtype. Trainer placement uses the live trainer dtype."""

    double_buffer_weights: bool = True
    """Dedicated placement only: load updates into an inactive model and atomically swap."""


class ValueCheckpointConfig(BaseConfig):
    interval: int | None = Field(None, ge=1)
    """Save full value training state every N value updates."""

    resume_step: int | None = Field(None, ge=-1)
    """Value update to resume; -1 selects the latest value checkpoint."""

    keep_last: int | None = Field(2, ge=1)


def validate_value_model_capabilities(model: ModelConfig) -> None:
    if model.lora is not None:
        raise ValueError("value functions do not support LoRA yet")
    if model.vlm is not None:
        raise ValueError("value functions do not support VLM training yet")
    if model.cp != 1:
        raise ValueError("value functions do not support context parallelism yet")


class ValueFunctionConfig(BaseConfig):
    model: ModelConfig | None = None
    """Value backbone. None copies the policy trainer model configuration."""

    tokenizer_name: str | None = None
    """Tokenizer vocabulary expected by the value backbone. Defaults to ``model.name`` and must match the policy tokenizer."""

    loss: ValueLossConfig = ClassificationValueLossConfig()
    """Value-head objective. Defaults to two-bin classification over ``[0, 1]``."""

    optim: OptimizerConfig = AdamWConfig(lr=1e-5)

    scheduler: SchedulerConfig = ConstantSchedulerConfig()

    gamma: float = Field(1.0, ge=0, le=1)

    gae_lambda: float = Field(1.0, ge=0, le=1)
    """Lambda used for the policy's generalized advantage estimate."""

    value_target_lambda: float = Field(1.0, ge=0, le=1)
    """Independent lambda used for the critic's TD(lambda) return target."""

    batch_size: int | None = Field(None, ge=1)
    """Rollouts per critic optimizer batch. None inherits the orchestrator rollout batch size."""

    updates_per_batch: int = Field(1, ge=1)
    """Optimizer updates on each post-warmup rollout batch before it is discarded."""

    warmup_updates: int = Field(0, ge=0)
    """Evaluator value version required before the first policy batch ships."""

    transport: LatestZMQValueTransportConfig = LatestZMQValueTransportConfig()

    weight_broadcast: NCCLValueWeightBroadcastConfig = NCCLValueWeightBroadcastConfig()

    evaluator: ValueEvaluatorConfig = ValueEvaluatorConfig()

    ckpt: ValueCheckpointConfig | None = ValueCheckpointConfig()

    output_dir: Path = Path("outputs/value")

    max_steps: int | None = Field(None, ge=1)
    """Optional independent cap on value optimizer updates."""

    matmul_precision: Literal["highest", "high", "medium"] = "high"

    dist_timeout_seconds: int = Field(3600, ge=1)

    log: TrainerLogConfig = TrainerLogConfig()

    wandb: WandbConfig | None = None

    env_vars: EnvVars = {}

    @model_validator(mode="after")
    def validate_initial_scope(self):
        if self.max_steps is not None and self.warmup_updates > self.max_steps:
            raise ValueError("value_function.warmup_updates cannot exceed value_function.max_steps")
        dedicated = self.evaluator.placement == "dedicated"
        ports = [self.transport.port, self.evaluator.port]
        if dedicated:
            ports.extend([self.weight_broadcast.port, self.weight_broadcast.control_port])
        if len(set(ports)) != len(ports):
            if dedicated:
                raise ValueError("value transport, weight, weight-control, and evaluator ports must be distinct")
            raise ValueError("value transport and evaluator ports must be distinct")
        endpoint_keys: set[tuple[str, int]] = set()
        reserved_local_ports = {self.transport.port}
        if dedicated:
            reserved_local_ports.update([self.weight_broadcast.port, self.weight_broadcast.control_port])
        local_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
        for endpoint in self.evaluator.base_url:
            parsed = urlparse(endpoint)
            if parsed.hostname is None or parsed.port is None:
                raise ValueError(f"value evaluator base_url must include a host and explicit port: {endpoint}")
            key = (parsed.hostname, parsed.port)
            if key in endpoint_keys:
                raise ValueError(f"duplicate value evaluator endpoint: {endpoint}")
            endpoint_keys.add(key)
            if parsed.hostname in local_hosts and parsed.port in reserved_local_ports:
                reserved_by = "value transport or weight broadcast" if dedicated else "value transport"
                raise ValueError(f"local value evaluator port {parsed.port} conflicts with {reserved_by}")
        if not dedicated:
            if len(self.evaluator.base_url) != 1:
                raise ValueError("trainer-placed value evaluation requires exactly one evaluator base_url")
            endpoint_port = urlparse(self.evaluator.base_url[0]).port
            if endpoint_port != self.evaluator.port:
                raise ValueError("trainer-placed evaluator base_url port must equal value_function.evaluator.port")
            if self.weight_broadcast.evaluator_world_size != 1:
                raise ValueError("trainer-placed value evaluation supports exactly one evaluator endpoint")
        if self.model is None:
            return self
        validate_value_model_capabilities(self.model)
        return self
