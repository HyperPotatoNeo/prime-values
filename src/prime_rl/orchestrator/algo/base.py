"""The per-env algorithm runtime: the :class:`Algorithm` base class.

Each named class in this package *is* one training algorithm, one module per
algorithm: it owns the algorithm's two scoring hooks directly —
``score_rollout`` (per arrival) and ``score_group`` (per group) — and declares
which loss component its action tokens feed (``action_loss_type``). Reading a
module top to bottom reads the algorithm; writing your own is subclassing
:class:`Algorithm` and overriding the hooks its signal needs. Shared rollout-credit
math (group baselines and GAE) lives as plain functions in
``advantage.py``;
duplication of orchestration between similar algorithms (e.g. OPD and OPSD) is
accepted so each module stays self-contained.

The two hooks are one scope-and-timing ladder — the wider scope is unlocked by
a later barrier, so the two axes coincide. Both are ``async`` (either may do
I/O); a hook that only does advantage math never awaits:

- ``score_rollout(rollout)`` — one rollout, on arrival: rollout-local signals
  (raw reward, process rewards, echo's observation weighting) *and* per-rollout
  I/O against another model — an inference pool the algorithm connected in
  ``setup()`` (a frozen teacher) or the live policy (opsd's self-distillation),
  queried with bounded concurrency. No siblings.
- ``score_group(group)`` — the cohort, on group completion, *before* filtering
  (filters read the streams): group-relative credit (GRPO/MaxRL baselines).

How rollouts are *produced* is not the algorithm's concern: that is the env's
:class:`~prime_rl.orchestrator.sampler.Sampler`, and sample construction
(interleaving, with observation-token provenance via structural node
attribution) is pure pipeline.

The pipeline (train sink) drives each algorithm through its non-virtual
:meth:`Algorithm.finalize_rollout` / :meth:`Algorithm.finalize_group` methods
and reads the class declarations; it never branches on algorithm config fields
or model roles — liveness of a reference is the only runtime distinction.
prime-rl hosts exactly one model — the trainable policy, whose pool every
algorithm is handed (``self.policy_pool``): use it for anything that scores
against the live model (opsd's self-distillation teacher, an LLM judge, ...).
Every *frozen* model an algorithm needs is an external endpoint it *connects to*
(never launches) and owns, in :meth:`Algorithm.setup`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from prime_rl.configs.algorithm import ActionLossType, AlgoConfig, FrozenModelConfig, GRPOAlgoConfig
from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.orchestrator.algo.advantage import compute_gae
from prime_rl.orchestrator.algo.routing import stamp_advantages, stamp_loss_routing
from prime_rl.orchestrator.trajectories import TrainingLayout
from prime_rl.orchestrator.value_context import TokenPrefix
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from renderers import RendererConfig

    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool
    from prime_rl.value.client import ValueEvaluatorClient


async def connect_frozen_pool(
    config: FrozenModelConfig, *, renderer_config: RendererConfig | None = None
) -> InferencePool:
    """Connect a client pool to an inline frozen model and wait for it to be
    ready. The endpoint is externally hosted — prime-rl connects and waits,
    never launches.

    When ``renderer_config`` is set, the pool's train client is the renderer
    (token-in/out) client — required when the frozen model *generates* rollouts
    (sft), so the rollout carries tokens. Left as plain chat-completions
    otherwise (opd/opsd read teacher logprobs via prefill, where the train
    client type is moot)."""
    from prime_rl.utils.client import setup_inference_pool

    get_logger().info(f"Initializing frozen model pool (model={config.name}, base_url={', '.join(config.base_url)})")
    if renderer_config is not None:
        pool = await setup_inference_pool(
            config, model_name=config.name, train_client_type="renderer", renderer_config=renderer_config
        )
    else:
        pool = await setup_inference_pool(config, model_name=config.name)
    await pool.wait_for_ready(config.name)
    return pool


class Algorithm:
    """Base class for one env's training algorithm — the runtime of the
    algorithm config's per-token training signal (its sibling :class:`Sampler`
    interprets the ``sampling`` half).

    Everything on this class is yours to override; the pipeline drives the
    compilation through the non-virtual :meth:`finalize_rollout` /
    :meth:`finalize_group` methods and never calls anything else. The surface is:

    - declarations — which loss component the action tokens feed
      (``action_loss_type``);
    - lifecycle — :meth:`setup` connects client pools to the frozen models
      the algorithm declares, resolving each reference via :meth:`connect`;
    - the two scoring hooks, each ``async`` and given the :class:`Rollout`
      directly — read the trace, write credit via
      :meth:`Rollout.assign_advantages`. They are
      async so either stage may do I/O — e.g. a process-reward model or a
      teacher at arrival, or a judge at group time whose signal a pre-batch
      filter then reads; a hook that only does advantage math simply never
      awaits.

      - :meth:`score_rollout` — one rollout, on arrival: rollout-local credit,
        observation ce weights, or per-token results from a model the algorithm
        connected in :meth:`setup` (e.g. teacher reference logprobs). Default:
        nothing.
      - :meth:`score_group` — the cohort, *before* filtering (filters read the
        streams): group-relative credit. Default: nothing — rollouts keep
        ``advantages=None``, so advantage-based filters skip them.

    Rollout-local model I/O lives in :meth:`score_rollout` and runs at arrival,
    *before* pre-batch filters. Group-conditioned value evaluation waits for
    :meth:`finalize_group`, when every sibling trajectory is available. Both
    paths therefore pay compute on rollouts that may later be filtered.

    Constructed with the algorithm config it interprets plus the live policy
    pool (``self.policy_pool`` — always available, never closed by the
    algorithm). An algorithm that needs to tokenize (e.g. opsd's demonstration
    hint) builds its own renderer in :meth:`setup` from its config; the policy's
    renderer is not threaded in."""

    action_loss_type: ClassVar[ActionLossType] = "rl"

    @property
    def minimum_group_size(self) -> int:
        """Smallest surviving rollout group this algorithm can score."""
        return 1

    def __init__(
        self,
        config: AlgoConfig,
        policy_pool: InferencePool,
        *,
        value_evaluator: ValueEvaluatorClient | None = None,
        value_config: ValueFunctionConfig | None = None,
    ):
        self.config = config
        self.policy_pool = policy_pool
        self.connected_pools: list[InferencePool] = []  # frozen pools connected in setup(); closed at shutdown
        self.value_evaluator = value_evaluator
        self.value_config = value_config
        if (value_evaluator is None) != (value_config is None):
            raise ValueError("value evaluator and value config must be provided together")

    async def setup(self) -> None:
        """Connect client pools to the algorithm's frozen models — override
        and resolve each reference via :meth:`connect`. The base has nothing
        to connect."""

    def metrics(self) -> dict[str, float]:
        """Current algorithm-local metrics, without an environment prefix."""
        return {}

    def metric_keys(self) -> list[str]:
        """Stable metric names, including keys not yet present in ``metrics``."""
        return list(self.metrics())

    def state_dict(self) -> dict[str, Any]:
        """Small orchestrator-owned state persisted with policy checkpoints."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore orchestrator-owned state, rejecting it for stateless algorithms."""
        if state:
            raise ValueError(f"{type(self).__name__} is stateless but the checkpoint contains algorithm state")

    async def connect(self, reference: FrozenModelConfig) -> InferencePool:
        """Connect a client pool to a frozen model endpoint and track it in
        ``connected_pools`` — the host closes what the algorithm opened, at
        shutdown. The live policy is never connected here; opsd receives the
        policy pool directly."""
        pool = await connect_frozen_pool(reference)
        self.connected_pools.append(pool)
        return pool

    async def score_rollout(self, rollout: Rollout) -> None:
        """Arrival phase, one rollout, before its group is complete: write
        rollout-local credit (``rollout.assign_advantages``), observation ce
        weights (echo), or per-token results from a model — an inference pool
        connected in :meth:`setup`, or the live policy (opsd). No siblings, no
        group stats."""

    async def score_group(self, group: list[Rollout]) -> None:
        """Group phase, the finalized cohort, before filtering: write
        group-relative credit."""

    async def finalize_rollout(self, rollout: Rollout) -> None:
        """Arrival phase (non-virtual): rollout-local scoring as each rollout is
        tokenized."""
        if rollout.samples:
            await self.score_rollout(rollout)
            if (
                self.value_evaluator is not None
                and self.value_config is not None
                and self.value_config.privileged_context != "group_leave_one_out"
            ):
                await self._evaluate_value(rollout)

    async def finalize_group(self, rollouts: list[Rollout]) -> None:
        """Group phase (non-virtual): group-relative scoring, then stamp each
        sample's wire fields (the advantage stream + loss routing). After this
        the records are frozen — groups die at stamping."""
        if self.value_evaluator is not None:
            assert self.value_config is not None
            group_conditioned = self.value_config.privileged_context == "group_leave_one_out"
            if group_conditioned and any(rollout.value_prefix is None for rollout in rollouts):
                raise RuntimeError("group_leave_one_out value context was not attached before group finalization")
            if group_conditioned:
                # Peer context first becomes available here. Task-conditioned
                # rollouts retain their arrival-time values and versions.
                await self._evaluate_value_group(rollouts)
        await self.score_group(rollouts)
        for rollout in rollouts:
            stamp_advantages(rollout)
            for sample in rollout.samples:
                stamp_loss_routing(sample, self.action_loss_type)

    async def _evaluate_value_group(self, rollouts: list[Rollout]) -> None:
        assert self.value_evaluator is not None
        assert self.value_config is not None
        layouts = [self._sequential_value_layout(rollout) for rollout in rollouts]
        token_ids = [
            self._value_input(sample.token_ids, rollout.value_prefix)
            for rollout in rollouts
            for sample in rollout.samples
        ]
        response = await self.value_evaluator.evaluate(token_ids)
        if len(response.values) != len(token_ids):
            raise ValueError(
                f"value evaluator returned {len(response.values)} branches for group request with {len(token_ids)}"
            )

        updates = []
        offset = 0
        for rollout, layout in zip(rollouts, layouts, strict=True):
            count = len(rollout.samples)
            updates.append(
                self._compute_value_result(
                    rollout,
                    response.values[offset : offset + count],
                    layout=layout,
                )
            )
            offset += count
        for rollout, (predictions, advantages, returns) in zip(rollouts, updates, strict=True):
            rollout.value_predictions = predictions
            rollout.value_advantages = advantages
            rollout.value_returns = returns
            rollout.value_version = response.version

    async def _evaluate_value(self, rollout: Rollout) -> None:
        assert self.value_evaluator is not None
        layout = self._sequential_value_layout(rollout)
        response = await self.value_evaluator.evaluate(
            [self._value_input(sample.token_ids, rollout.value_prefix) for sample in rollout.samples]
        )
        self._assign_value_result(rollout, response.values, response.version, layout=layout)

    def _value_input(self, token_ids: list[int], prefix: TokenPrefix | None = None) -> list[int]:
        assert self.value_config is not None and self.value_config.model is not None
        if prefix is not None:
            return prefix.apply(token_ids)
        return token_ids[: self.value_config.model.seq_len]

    def _sequential_value_layout(self, rollout: Rollout) -> TrainingLayout | None:
        """Return generation order when configured credit must cross branches."""
        assert self.value_config is not None
        config = cast(GRPOAlgoConfig, self.config)
        needs_stitching = (
            self.value_config.gamma < 1.0
            or self.value_config.value_target_lambda < 1.0
            or (config.baseline.uses_policy_gae and self.value_config.gae_lambda < 1.0)
        )
        if len(rollout.samples) <= 1 or not needs_stitching:
            return None
        if config.branch_semantics != "sequential":
            raise ValueError(
                "temporal value credit over multiple branches requires "
                "algo.branch_semantics='sequential'; "
                f"env={rollout.env_name!r}, group={rollout.group_id}, "
                f"branches={len(rollout.samples)}, gamma={self.value_config.gamma}, "
                f"gae_lambda={self.value_config.gae_lambda}, "
                f"value_target_lambda={self.value_config.value_target_lambda}"
            )
        layout = rollout.training_layout
        assert layout is not None
        assert self.value_config.model is not None
        if rollout.value_prefix is not None:
            return layout
        visible_lengths = [min(len(sample.token_ids), self.value_config.model.seq_len) for sample in rollout.samples]
        for span in layout.trainable_spans:
            end = span.sample_start + span.length
            visible_length = visible_lengths[span.sample_index]
            if end > visible_length:
                raise ValueError(
                    "sequential temporal credit requires every trainable token to be visible "
                    f"to the critic; sample {span.sample_index} span [{span.sample_start}, {end}) "
                    f"exceeds visible length {visible_length}"
                )
        return layout

    def _assign_value_result(
        self,
        rollout: Rollout,
        predictions: list[list[float]],
        version: int,
        *,
        layout: TrainingLayout | None = None,
    ) -> None:
        value_predictions, advantages, returns = self._compute_value_result(
            rollout,
            predictions,
            layout=layout,
        )
        rollout.value_predictions = value_predictions
        rollout.value_advantages = advantages
        rollout.value_returns = returns
        rollout.value_version = version

    def _compute_value_result(
        self,
        rollout: Rollout,
        predictions: list[list[float]],
        *,
        layout: TrainingLayout | None = None,
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        assert self.value_config is not None
        if len(predictions) != len(rollout.samples):
            raise ValueError(
                f"value evaluator returned {len(predictions)} branches for rollout with {len(rollout.samples)} samples"
            )
        if layout is None:
            layout = self._sequential_value_layout(rollout)
        projected_predictions: list[list[float]] = []
        for sample, values in zip(rollout.samples, predictions, strict=True):
            value_input = self._value_input(sample.token_ids, rollout.value_prefix)
            if len(values) != len(value_input):
                if rollout.value_prefix is None:
                    raise ValueError(
                        f"value evaluator returned {len(values)} tokens for truncated value input of {len(value_input)}"
                    )
                raise ValueError(
                    f"value evaluator returned {len(values)} tokens for conditioned value input of {len(value_input)}"
                )
            projected = rollout.value_prefix.project(values) if rollout.value_prefix is not None else values
            projected_predictions.append(projected)

        if layout is None:
            advantages: list[list[float]] = []
            returns: list[list[float]] = []
            for sample, projected in zip(rollout.samples, projected_predictions, strict=True):
                value_length = len(projected)
                sample_advantages, sample_returns = compute_gae(
                    reward=float(rollout.reward),
                    values=projected,
                    mask=sample.mask[:value_length],
                    gamma=self.value_config.gamma,
                    gae_lambda=self.value_config.gae_lambda,
                    value_target_lambda=self.value_config.value_target_lambda,
                )
                padding = len(sample.token_ids) - value_length
                advantages.append(sample_advantages + [0.0] * padding)
                returns.append(sample_returns + [0.0] * padding)
        else:
            flat_values = [
                value
                for span in layout.trainable_spans
                for value in projected_predictions[span.sample_index][
                    span.sample_start : span.sample_start + span.length
                ]
            ]
            flat_advantages, flat_returns = compute_gae(
                reward=float(rollout.reward),
                values=flat_values,
                mask=[True] * len(flat_values),
                gamma=self.value_config.gamma,
                gae_lambda=self.value_config.gae_lambda,
                value_target_lambda=self.value_config.value_target_lambda,
            )
            advantages = [[0.0] * len(sample.token_ids) for sample in rollout.samples]
            returns = [[0.0] * len(sample.token_ids) for sample in rollout.samples]
            offset = 0
            for span in layout.trainable_spans:
                end = offset + span.length
                sample_end = span.sample_start + span.length
                advantages[span.sample_index][span.sample_start : sample_end] = flat_advantages[offset:end]
                returns[span.sample_index][span.sample_start : sample_end] = flat_returns[offset:end]
                offset = end
        value_predictions = [
            values + [0.0] * (len(sample.token_ids) - len(values))
            for sample, values in zip(rollout.samples, projected_predictions, strict=True)
        ]
        return value_predictions, advantages, returns
