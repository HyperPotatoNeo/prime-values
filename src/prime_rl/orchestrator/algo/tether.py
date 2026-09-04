"""Adaptive mixed baseline for a fixed critic lambda-return."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from prime_rl.configs.algorithm import AdaptiveTetherConfig, TetherBaselineConfig
from prime_rl.orchestrator.algo.advantage import group_baselines
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


@dataclass(frozen=True)
class TetherRegressionStats:
    """Token-weighted sufficient statistics for ``Q_lambda-B ~= rho * (V-B)``."""

    weight: int = 0
    feature_feature: float = 0.0
    feature_target: float = 0.0
    target_target: float = 0.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("invalid adaptive TETHER regression count")
        moments = (self.feature_feature, self.feature_target, self.target_target)
        if not all(math.isfinite(value) for value in moments):
            raise ValueError("adaptive TETHER regression moments must be finite")
        if self.feature_feature < 0 or self.target_target < 0:
            raise ValueError("adaptive TETHER diagonal moments must be non-negative")

    def __add__(self, other: TetherRegressionStats) -> TetherRegressionStats:
        return TetherRegressionStats(
            weight=self.weight + other.weight,
            feature_feature=self.feature_feature + other.feature_feature,
            feature_target=self.feature_target + other.feature_target,
            target_target=self.target_target + other.target_target,
        )


@dataclass(frozen=True)
class TetherCoefficientTable:
    """Piecewise-constant rho values over causal branch-local action position."""

    rho: tuple[float, ...]
    bin_size: int

    def __post_init__(self) -> None:
        if len(self.rho) < 2 or self.bin_size < 1:
            raise ValueError("positioned TETHER needs at least two positive-width bins")
        if not all(math.isfinite(value) for value in self.rho):
            raise ValueError("positioned TETHER coefficients must be finite")

    @property
    def num_bins(self) -> int:
        return len(self.rho)

    def bin_index(self, action_position: int) -> int:
        if action_position < 0:
            raise ValueError("TETHER action position must be non-negative")
        return min(action_position // self.bin_size, self.num_bins - 1)


@dataclass(frozen=True)
class TetherRolloutStats:
    """One rollout's sufficient statistics partitioned by position bin."""

    bins: tuple[TetherRegressionStats, ...]

    def __post_init__(self) -> None:
        if len(self.bins) < 2:
            raise ValueError("positioned TETHER rollout stats need at least two bins")


@dataclass
class _StatsAccumulator:
    weight: int = 0
    feature_feature: float = 0.0
    feature_target: float = 0.0
    target_target: float = 0.0

    def add_row(self, feature: float, target: float) -> None:
        self.weight += 1
        self.feature_feature += feature * feature
        self.feature_target += feature * target
        self.target_target += target * target

    def freeze(self) -> TetherRegressionStats:
        return TetherRegressionStats(**asdict(self))


def _fit_tether(stats: TetherRegressionStats, *, ridge: float) -> float | None:
    """Solve the token-normalized, relative-ridge scalar regression."""
    if stats.weight < 1:
        raise ValueError("adaptive TETHER fit needs at least one row")
    inv_weight = 1.0 / stats.weight
    feature_power = stats.feature_feature * inv_weight
    target_power = stats.target_target * inv_weight
    scale = max(feature_power, target_power)
    denominator = feature_power + ridge * scale
    if (
        feature_power <= 0.0
        or scale <= 0.0
        or not all(math.isfinite(value) for value in (feature_power, target_power, denominator))
        or denominator <= 0.0
    ):
        return None
    rho = stats.feature_target * inv_weight / denominator
    return rho if math.isfinite(rho) else None


def _residual_sum(stats: TetherRegressionStats, rho: float) -> float:
    return max(
        stats.target_target - 2.0 * rho * stats.feature_target + rho * rho * stats.feature_feature,
        0.0,
    )


class AdaptiveTetherCoefficient:
    """Lagged exact-rollout-window ridge fits with an uncorrected coefficient EMA."""

    def __init__(
        self,
        config: AdaptiveTetherConfig,
        *,
        batch_size: int,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("adaptive TETHER batch_size must be positive")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("adaptive TETHER gamma must be in [0, 1]")
        if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("adaptive TETHER gae_lambda must be in [0, 1]")
        self.config = config
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rho = config.initial_rho
        self.pending_rollouts = 0
        self.pending = TetherRegressionStats()
        self.updates = 0
        self.skipped_updates = 0
        self.last_fit_rho = self.rho
        self.last_fit_valid = False
        self.last_mse_group = 0.0
        self.last_mse_fit = 0.0
        self.last_mse_ema = 0.0
        self.last_feature_power = 0.0

    def observe_group(self, stats: list[TetherRegressionStats]) -> None:
        """Queue a fully scored group, consuming exact rollout-count windows."""
        for rollout_stats in stats:
            self.pending += rollout_stats
            self.pending_rollouts += 1
            if self.pending_rollouts == self.batch_size:
                self._update(self.pending)
                self.pending_rollouts = 0
                self.pending = TetherRegressionStats()

    def _update(self, stats: TetherRegressionStats) -> None:
        self.last_fit_valid = False
        self.last_fit_rho = self.rho
        if stats.weight == 0:
            self.last_mse_group = 0.0
            self.last_mse_fit = 0.0
            self.last_mse_ema = 0.0
            self.last_feature_power = 0.0
            self._skip("no trainable tokens")
            return

        inv_weight = 1.0 / stats.weight
        feature_power = stats.feature_feature * inv_weight
        target_power = stats.target_target * inv_weight
        self.last_mse_group = target_power
        self.last_feature_power = feature_power
        self.last_mse_fit = self.last_mse_ema = _residual_sum(stats, self.rho) * inv_weight
        fit_rho = _fit_tether(stats, ridge=self.config.ridge)
        if fit_rho is None:
            self._skip("degenerate or non-finite fit")
            return

        decay = self.config.ema_decay
        self.rho = decay * self.rho + (1.0 - decay) * fit_rho
        self.updates += 1
        self.last_fit_rho = fit_rho
        self.last_fit_valid = True
        self.last_mse_fit = _residual_sum(stats, fit_rho) * inv_weight
        self.last_mse_ema = _residual_sum(stats, self.rho) * inv_weight
        get_logger().info(
            f"Adaptive TETHER fit {self.updates} | rho={self.rho:.4f} | "
            f"batch fit={fit_rho:.4f} | mse group={self.last_mse_group:.6g}, "
            f"fit={self.last_mse_fit:.6g}"
        )

    def _skip(self, reason: str) -> None:
        self.skipped_updates += 1
        get_logger().warning(f"Skipping adaptive TETHER window with {reason}")

    def metrics(self) -> dict[str, float]:
        return {
            "tether/rho": self.rho,
            "tether/batch_fit_rho": self.last_fit_rho,
            "tether/batch_fit_valid": float(self.last_fit_valid),
            "tether/updates": float(self.updates),
            "tether/skipped_updates": float(self.skipped_updates),
            "tether/pending_rollouts": float(self.pending_rollouts),
            "tether/regression_batch_size": float(self.batch_size),
            "tether/mse_group": self.last_mse_group,
            "tether/mse_batch_fit": self.last_mse_fit,
            "tether/mse_post_fit_ema": self.last_mse_ema,
            "tether/feature_power": self.last_feature_power,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "kind": "adaptive_tether",
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ridge": self.config.ridge,
            "ema_decay": self.config.ema_decay,
            "rho": self.rho,
            "pending_rollouts": self.pending_rollouts,
            "pending": asdict(self.pending),
            "updates": self.updates,
            "skipped_updates": self.skipped_updates,
            "last_fit_rho": self.last_fit_rho,
            "last_fit_valid": self.last_fit_valid,
            "last_mse_group": self.last_mse_group,
            "last_mse_fit": self.last_mse_fit,
            "last_mse_ema": self.last_mse_ema,
            "last_feature_power": self.last_feature_power,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        schema_version = _strict_int(state.get("schema_version"), "schema_version", minimum=1)
        if schema_version != 2:
            raise ValueError(f"unsupported adaptive TETHER checkpoint schema {schema_version}")
        expected = (
            "adaptive_tether",
            self.batch_size,
            self.gamma,
            self.gae_lambda,
            self.config.ridge,
            self.config.ema_decay,
        )
        saved = (
            state.get("kind"),
            _strict_int(state.get("batch_size"), "batch_size", minimum=1),
            _finite_float(state.get("gamma"), "gamma"),
            _finite_float(state.get("gae_lambda"), "gae_lambda"),
            _finite_float(state.get("ridge"), "ridge"),
            _finite_float(state.get("ema_decay"), "ema_decay"),
        )
        if saved != expected:
            raise ValueError(f"adaptive TETHER checkpoint contract {saved} does not match config {expected}")
        self.rho = _finite_float(state.get("rho"), "rho")
        self.pending_rollouts = _strict_int(state.get("pending_rollouts"), "pending_rollouts")
        self.pending = TetherRegressionStats(**_strict_dict(state.get("pending"), "pending"))
        self.updates = _strict_int(state.get("updates"), "updates")
        self.skipped_updates = _strict_int(state.get("skipped_updates"), "skipped_updates")
        self.last_fit_rho = _finite_float(state.get("last_fit_rho"), "last_fit_rho")
        self.last_fit_valid = _strict_bool(state.get("last_fit_valid"), "last_fit_valid")
        self.last_mse_group = _nonnegative_float(state.get("last_mse_group"), "last_mse_group")
        self.last_mse_fit = _nonnegative_float(state.get("last_mse_fit"), "last_mse_fit")
        self.last_mse_ema = _nonnegative_float(state.get("last_mse_ema"), "last_mse_ema")
        self.last_feature_power = _nonnegative_float(state.get("last_feature_power"), "last_feature_power")
        if self.pending_rollouts >= self.batch_size:
            raise ValueError("adaptive TETHER checkpoint has an invalid pending rollout count")


class AdaptivePositionTetherCoefficients:
    """Lagged independent rho fits over fixed causal action-position bins."""

    def __init__(
        self,
        config: AdaptiveTetherConfig,
        *,
        batch_size: int,
        gamma: float,
        gae_lambda: float,
        num_bins: int,
        bin_size: int,
        min_bin_rollouts: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("adaptive positioned TETHER batch_size must be positive")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("adaptive positioned TETHER gamma must be in [0, 1]")
        if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("adaptive positioned TETHER gae_lambda must be in [0, 1]")
        if num_bins < 2 or bin_size < 1:
            raise ValueError("adaptive positioned TETHER needs at least two positive-width bins")
        if not 1 <= min_bin_rollouts <= batch_size:
            raise ValueError("adaptive positioned TETHER has invalid minimum bin support")
        self.config = config
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_bins = num_bins
        self.bin_size = bin_size
        self.min_bin_rollouts = min_bin_rollouts
        self._rho = [config.initial_rho] * num_bins
        self.pending_rollouts = 0
        self.pending_bins = [TetherRegressionStats() for _ in range(num_bins)]
        self.pending_contributors = [0] * num_bins
        self.updates = 0
        self.skipped_updates = 0
        self.bin_updates = [0] * num_bins
        self.last_fit_valid = [False] * num_bins
        self.last_contributors = [0] * num_bins
        self.last_mse_group = 0.0
        self.last_mse_fit = 0.0
        self.last_mse_ema = 0.0
        self.last_feature_power = 0.0
        self.last_fit_token_fraction = 0.0

    @property
    def coefficient_table(self) -> TetherCoefficientTable:
        return TetherCoefficientTable(tuple(self._rho), self.bin_size)

    def observe_group(self, stats: list[TetherRolloutStats]) -> None:
        """Queue a fully scored group, consuming exact rollout-count windows."""
        for rollout_stats in stats:
            if len(rollout_stats.bins) != self.num_bins:
                raise ValueError("positioned TETHER rollout bin count mismatch")
            for index, bin_stats in enumerate(rollout_stats.bins):
                self.pending_bins[index] += bin_stats
                self.pending_contributors[index] += int(bin_stats.weight > 0)
            self.pending_rollouts += 1
            if self.pending_rollouts == self.batch_size:
                self._update(self.pending_bins, self.pending_contributors)
                self.pending_rollouts = 0
                self.pending_bins = [TetherRegressionStats() for _ in range(self.num_bins)]
                self.pending_contributors = [0] * self.num_bins

    def _update(self, stats: list[TetherRegressionStats], contributors: list[int]) -> None:
        total_weight = sum(item.weight for item in stats)
        self.last_fit_valid = [False] * self.num_bins
        self.last_contributors = list(contributors)
        if total_weight == 0:
            self.skipped_updates += 1
            self.last_mse_group = 0.0
            self.last_mse_fit = 0.0
            self.last_mse_ema = 0.0
            self.last_feature_power = 0.0
            self.last_fit_token_fraction = 0.0
            get_logger().warning("Skipping adaptive positioned TETHER window with no trainable tokens")
            return

        fit_weight = 0
        fit_residual = 0.0
        for index, (bin_stats, support) in enumerate(zip(stats, contributors, strict=True)):
            if bin_stats.weight == 0 or support < self.min_bin_rollouts:
                continue
            fit_rho = _fit_tether(bin_stats, ridge=self.config.ridge)
            if fit_rho is None:
                continue
            decay = self.config.ema_decay
            self._rho[index] = decay * self._rho[index] + (1.0 - decay) * fit_rho
            self.last_fit_valid[index] = True
            self.bin_updates[index] += 1
            fit_weight += bin_stats.weight
            fit_residual += _residual_sum(bin_stats, fit_rho)

        if any(self.last_fit_valid):
            self.updates += 1
        else:
            self.skipped_updates += 1
        self.last_mse_group = sum(item.target_target for item in stats) / total_weight
        self.last_mse_fit = fit_residual / fit_weight if fit_weight else 0.0
        self.last_mse_ema = (
            sum(_residual_sum(item, rho) for item, rho in zip(stats, self._rho, strict=True)) / total_weight
        )
        self.last_feature_power = sum(item.feature_feature for item in stats) / total_weight
        self.last_fit_token_fraction = fit_weight / total_weight
        get_logger().info(
            f"Adaptive positioned TETHER window | "
            f"updated={sum(self.last_fit_valid)}/{self.num_bins} bins | "
            f"rho=[{min(self._rho):.4f}, {max(self._rho):.4f}]"
        )

    def metrics(self) -> dict[str, float]:
        metrics = {
            "tether/updates": float(self.updates),
            "tether/skipped_updates": float(self.skipped_updates),
            "tether/pending_rollouts": float(self.pending_rollouts),
            "tether/regression_batch_size": float(self.batch_size),
            "tether/mse_group": self.last_mse_group,
            "tether/mse_batch_fit": self.last_mse_fit,
            "tether/mse_post_fit_ema": self.last_mse_ema,
            "tether/feature_power": self.last_feature_power,
            "tether/fit_token_fraction": self.last_fit_token_fraction,
            "tether/position/num_bins": float(self.num_bins),
            "tether/position/fitted_bins": float(sum(self.last_fit_valid)),
            "tether/position/ever_fitted_bins": float(sum(count > 0 for count in self.bin_updates)),
            "tether/position/min_bin_rollouts": float(self.min_bin_rollouts),
            "tether/position/rho_min": min(self._rho),
            "tether/position/rho_max": max(self._rho),
        }
        for index in range(self.num_bins):
            prefix = f"tether/position/bin_{index:03d}"
            metrics |= {
                f"{prefix}/rho": self._rho[index],
                f"{prefix}/fit_valid": float(self.last_fit_valid[index]),
                f"{prefix}/contributing_rollouts": float(self.last_contributors[index]),
            }
        return metrics

    def metric_keys(self) -> list[str]:
        return list(self.metrics())

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "adaptive_tether_position",
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ridge": self.config.ridge,
            "ema_decay": self.config.ema_decay,
            "num_bins": self.num_bins,
            "bin_size": self.bin_size,
            "min_bin_rollouts": self.min_bin_rollouts,
            "rho": list(self._rho),
            "pending_rollouts": self.pending_rollouts,
            "pending_bins": [asdict(item) for item in self.pending_bins],
            "pending_contributors": list(self.pending_contributors),
            "updates": self.updates,
            "skipped_updates": self.skipped_updates,
            "bin_updates": list(self.bin_updates),
            "last_fit_valid": list(self.last_fit_valid),
            "last_contributors": list(self.last_contributors),
            "last_mse_group": self.last_mse_group,
            "last_mse_fit": self.last_mse_fit,
            "last_mse_ema": self.last_mse_ema,
            "last_feature_power": self.last_feature_power,
            "last_fit_token_fraction": self.last_fit_token_fraction,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        schema_version = _strict_int(state.get("schema_version"), "schema_version", minimum=1)
        kind = state.get("kind")
        if schema_version != 1 or kind != "adaptive_tether_position":
            raise ValueError(
                f"adaptive positioned TETHER checkpoint kind/schema {(kind, schema_version)} is incompatible"
            )
        expected = (
            self.batch_size,
            self.gamma,
            self.gae_lambda,
            self.config.ridge,
            self.config.ema_decay,
            self.num_bins,
            self.bin_size,
            self.min_bin_rollouts,
        )
        saved = (
            _strict_int(state.get("batch_size"), "batch_size", minimum=1),
            _finite_float(state.get("gamma"), "gamma"),
            _finite_float(state.get("gae_lambda"), "gae_lambda"),
            _finite_float(state.get("ridge"), "ridge"),
            _finite_float(state.get("ema_decay"), "ema_decay"),
            _strict_int(state.get("num_bins"), "num_bins", minimum=2),
            _strict_int(state.get("bin_size"), "bin_size", minimum=1),
            _strict_int(state.get("min_bin_rollouts"), "min_bin_rollouts", minimum=1),
        )
        if saved != expected:
            raise ValueError(f"adaptive positioned TETHER checkpoint contract {saved} does not match {expected}")
        self._rho = _strict_float_vector(state.get("rho"), "rho", length=self.num_bins)
        self.pending_rollouts = _strict_int(state.get("pending_rollouts"), "pending_rollouts")
        pending_bins = state.get("pending_bins")
        if not isinstance(pending_bins, list) or len(pending_bins) != self.num_bins:
            raise ValueError("adaptive TETHER checkpoint contains invalid pending_bins")
        self.pending_bins = [TetherRegressionStats(**_strict_dict(item, "pending_bins")) for item in pending_bins]
        self.pending_contributors = _strict_int_vector(
            state.get("pending_contributors"),
            "pending_contributors",
            length=self.num_bins,
        )
        self.updates = _strict_int(state.get("updates"), "updates")
        self.skipped_updates = _strict_int(state.get("skipped_updates"), "skipped_updates")
        self.bin_updates = _strict_int_vector(state.get("bin_updates"), "bin_updates", length=self.num_bins)
        self.last_fit_valid = _strict_bool_vector(
            state.get("last_fit_valid"),
            "last_fit_valid",
            length=self.num_bins,
        )
        self.last_contributors = _strict_int_vector(
            state.get("last_contributors"),
            "last_contributors",
            length=self.num_bins,
        )
        self.last_mse_group = _nonnegative_float(state.get("last_mse_group"), "last_mse_group")
        self.last_mse_fit = _nonnegative_float(state.get("last_mse_fit"), "last_mse_fit")
        self.last_mse_ema = _nonnegative_float(state.get("last_mse_ema"), "last_mse_ema")
        self.last_feature_power = _nonnegative_float(
            state.get("last_feature_power"),
            "last_feature_power",
        )
        self.last_fit_token_fraction = _unit_float(
            state.get("last_fit_token_fraction"),
            "last_fit_token_fraction",
        )
        if self.pending_rollouts >= self.batch_size:
            raise ValueError("adaptive positioned TETHER checkpoint has an invalid pending rollout count")
        if any(value > self.pending_rollouts for value in self.pending_contributors):
            raise ValueError("adaptive positioned TETHER checkpoint has invalid pending bin support")
        for bin_stats, contributors in zip(self.pending_bins, self.pending_contributors, strict=True):
            if (
                (bin_stats.weight == 0) != (contributors == 0)
                or bin_stats.weight < contributors
                or (
                    bin_stats.weight == 0
                    and any(
                        value != 0.0
                        for value in (
                            bin_stats.feature_feature,
                            bin_stats.feature_target,
                            bin_stats.target_target,
                        )
                    )
                )
            ):
                raise ValueError("adaptive positioned TETHER checkpoint has inconsistent pending bin state")


class TetherRuntime:
    """Subtract a mixed group/value baseline from the critic policy lambda-return."""

    def __init__(
        self,
        config: TetherBaselineConfig,
        *,
        gamma: float,
        gae_lambda: float,
        value_seq_len: int,
        policy_seq_len: int,
        adaptive_batch_size: int | None,
        adaptive_min_value_version: int = 0,
    ) -> None:
        self.config = config
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_seq_len = value_seq_len
        self.policy_seq_len = policy_seq_len
        self.adaptive_min_value_version = adaptive_min_value_version
        self._applied_metrics = {"tether/mse_applied": 0.0, "tether/mse_group_applied": 0.0}
        self.adaptive: AdaptiveTetherCoefficient | None = None
        self.positioned_adaptive: AdaptivePositionTetherCoefficients | None = None
        if config.adaptive is not None:
            if adaptive_batch_size is None:
                raise ValueError("adaptive TETHER needs a resolved rollout batch size")
            position = config.adaptive.position
            if position is None:
                self.adaptive = AdaptiveTetherCoefficient(
                    config.adaptive,
                    batch_size=adaptive_batch_size,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )
            else:
                num_bins, bin_size, min_bin_rollouts = position.resolve(
                    policy_seq_len=policy_seq_len,
                    batch_size=adaptive_batch_size,
                )
                self.positioned_adaptive = AdaptivePositionTetherCoefficients(
                    config.adaptive,
                    batch_size=adaptive_batch_size,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    num_bins=num_bins,
                    bin_size=bin_size,
                    min_bin_rollouts=min_bin_rollouts,
                )

    @property
    def rho(self) -> float:
        if self.positioned_adaptive is not None:
            raise AttributeError("positioned TETHER has a coefficient table, not one rho")
        return self.adaptive.rho if self.adaptive is not None else self.config.rho

    def score_group(self, group: list[Rollout]) -> None:
        if self.positioned_adaptive is not None:
            self._score_positioned_group(group)
            return

        anchors = group_baselines([float(rollout.reward) for rollout in group], self.config.group)
        rho = self.rho
        regression_stats: list[TetherRegressionStats] = []
        for rollout, anchor in zip(group, anchors, strict=True):
            predictions = rollout.value_predictions
            raw_advantages = rollout.value_advantages
            if predictions is None or raw_advantages is None:
                raise RuntimeError("value evaluator did not attach TETHER predictions and policy GAE")
            if len(predictions) != len(rollout.samples) or len(raw_advantages) != len(rollout.samples):
                raise ValueError("TETHER value/sample branch count mismatch")

            rollout_advantages: list[float] = []
            weight = 0
            feature_feature = 0.0
            feature_target = 0.0
            target_target = 0.0
            for sample, values, value_advantages in zip(
                rollout.samples,
                predictions,
                raw_advantages,
                strict=True,
            ):
                sample_length = len(sample.token_ids)
                if len(values) != sample_length or len(value_advantages) != sample_length:
                    raise ValueError("TETHER value streams must span the padded sample stream")
                visible_length = self._critic_visible_length(rollout, len(sample.token_ids))
                if (self.gamma < 1.0 or self.gae_lambda < 1.0) and any(sample.mask[visible_length:]):
                    raise ValueError(
                        "TETHER temporal credit requires every trainable token to be visible to the critic"
                    )
                branch_advantages = [0.0] * sample_length
                for index in range(min(visible_length, self.policy_seq_len)):
                    if not sample.mask[index]:
                        continue
                    value_advantage = value_advantages[index]
                    feature = values[index] - anchor
                    centered_target = value_advantage + feature
                    branch_advantages[index] = value_advantage + (1.0 - rho) * feature
                    if self.adaptive is not None:
                        weight += 1
                        feature_feature += feature * feature
                        feature_target += feature * centered_target
                        target_target += centered_target * centered_target
                rollout_advantages.extend(branch_advantages)
            rollout.assign_advantages(rollout_advantages)

            if self.adaptive is not None:
                regression_stats.append(
                    TetherRegressionStats(
                        weight=weight,
                        feature_feature=feature_feature,
                        feature_target=feature_target,
                        target_target=target_target,
                    )
                )

        if self.adaptive is not None:
            self._record_applied_metrics((stats, rho) for stats in regression_stats)
            if self._adaptive_observation_ready(group):
                self.adaptive.observe_group(regression_stats)

    def _score_positioned_group(self, group: list[Rollout]) -> None:
        assert self.positioned_adaptive is not None
        anchors = group_baselines([float(rollout.reward) for rollout in group], self.config.group)
        coefficients = self.positioned_adaptive.coefficient_table
        regression_stats: list[TetherRolloutStats] = []
        for rollout, anchor in zip(group, anchors, strict=True):
            predictions = rollout.value_predictions
            raw_advantages = rollout.value_advantages
            if predictions is None or raw_advantages is None:
                raise RuntimeError("value evaluator did not attach TETHER predictions and policy GAE")
            branch_views = list(iter_trainable_branches(rollout))
            if (
                len(branch_views) != len(rollout.samples)
                or len(predictions) != len(rollout.samples)
                or len(raw_advantages) != len(rollout.samples)
            ):
                raise ValueError("positioned TETHER branch/sample/value alignment mismatch")

            rollout_advantages: list[float] = []
            rollout_bins = [TetherRegressionStats() for _ in range(coefficients.num_bins)]
            for (branch, train_mask), sample, values, value_advantages in zip(
                branch_views,
                rollout.samples,
                predictions,
                raw_advantages,
                strict=True,
            ):
                sample_length = len(sample.token_ids)
                if sample.token_ids != branch.token_ids or sample.mask != train_mask:
                    raise ValueError("positioned TETHER branch/sample token streams are misaligned")
                if len(values) != sample_length or len(value_advantages) != sample_length:
                    raise ValueError("TETHER value streams must span the padded sample stream")
                action_mask = branch.sampled_mask
                if len(action_mask) != sample_length or any(
                    trainable and not sampled for trainable, sampled in zip(train_mask, action_mask, strict=True)
                ):
                    raise ValueError("positioned TETHER train mask must follow native sampled provenance")
                visible_length = self._critic_visible_length(rollout, sample_length)
                if (self.gamma < 1.0 or self.gae_lambda < 1.0) and any(train_mask[visible_length:]):
                    raise ValueError(
                        "TETHER temporal credit requires every trainable token to be visible to the critic"
                    )

                branch_advantages = [0.0] * sample_length
                accumulators = [_StatsAccumulator() for _ in range(coefficients.num_bins)]
                action_position = 0
                for index in range(min(visible_length, self.policy_seq_len)):
                    if not action_mask[index]:
                        continue
                    bin_index = coefficients.bin_index(action_position)
                    action_position += 1
                    if not train_mask[index]:
                        continue
                    value_advantage = value_advantages[index]
                    feature = values[index] - anchor
                    centered_target = value_advantage + feature
                    rho = coefficients.rho[bin_index]
                    branch_advantages[index] = value_advantage + (1.0 - rho) * feature
                    accumulators[bin_index].add_row(feature, centered_target)
                rollout_advantages.extend(branch_advantages)
                rollout_bins = [
                    current + accumulator.freeze()
                    for current, accumulator in zip(rollout_bins, accumulators, strict=True)
                ]
            rollout.assign_advantages(rollout_advantages)
            regression_stats.append(TetherRolloutStats(tuple(rollout_bins)))

        self._record_applied_metrics(
            (stats, rho)
            for rollout_stats in regression_stats
            for stats, rho in zip(rollout_stats.bins, coefficients.rho, strict=True)
        )
        if self._adaptive_observation_ready(group):
            self.positioned_adaptive.observe_group(regression_stats)

    def _record_applied_metrics(self, rows: Iterable[tuple[TetherRegressionStats, float]]) -> None:
        """Measure the scored group with its applied coefficients before any fit."""
        weight = 0
        residual = 0.0
        group_residual = 0.0
        for stats, rho in rows:
            weight += stats.weight
            residual += _residual_sum(stats, rho)
            group_residual += stats.target_target
        self._applied_metrics = {
            "tether/mse_applied": residual / max(weight, 1),
            "tether/mse_group_applied": group_residual / max(weight, 1),
        }

    def _adaptive_observation_ready(self, group: list[Rollout]) -> bool:
        minimum = self.adaptive_min_value_version
        return minimum == 0 or all(
            rollout.value_version is not None and rollout.value_version >= minimum for rollout in group
        )

    def _critic_visible_length(self, rollout: Rollout, sample_length: int) -> int:
        critic_length = sample_length if rollout.value_prefix is not None else self.value_seq_len
        return min(sample_length, critic_length)

    def metrics(self) -> dict[str, float]:
        if self.positioned_adaptive is not None:
            return self.positioned_adaptive.metrics() | self._applied_metrics
        return self.adaptive.metrics() | self._applied_metrics if self.adaptive is not None else {}

    def metric_keys(self) -> list[str]:
        return list(self.metrics())

    def state_dict(self) -> dict[str, Any]:
        if self.positioned_adaptive is not None:
            return self.positioned_adaptive.state_dict()
        return self.adaptive.state_dict() if self.adaptive is not None else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.positioned_adaptive is not None:
            self.positioned_adaptive.load_state_dict(state)
            return
        if self.adaptive is None:
            raise ValueError("checkpoint contains adaptive TETHER state but adaptive mode is disabled")
        self.adaptive.load_state_dict(state)


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
    return value


def _finite_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    parsed = _finite_float(value, name)
    if parsed < 0.0:
        raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
    return parsed


def _unit_float(value: Any, name: str) -> float:
    parsed = _finite_float(value, name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
    return parsed


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
    return value


def _strict_float_vector(value: Any, name: str, *, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
    return [_finite_float(item, name) for item in value]


def _strict_int_vector(value: Any, name: str, *, length: int) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
    return [_strict_int(item, name) for item in value]


def _strict_bool_vector(value: Any, name: str, *, length: int) -> list[bool]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
    return [_strict_bool(item, name) for item in value]


def _strict_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
    return value
