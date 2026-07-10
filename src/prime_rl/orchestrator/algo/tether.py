"""TETHER baseline scoring and optional adaptive coefficient fitting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from prime_rl.configs.algorithm import MAX_TETHER_POSITION_BINS, AdaptiveTetherConfig, TetherBaselineConfig
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.utils.logger import get_logger
from prime_rl.value.math import group_baselines

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


@dataclass(frozen=True)
class TetherCoefficientTable:
    """Piecewise-constant alpha/rho values over causal action position."""

    alpha: tuple[float, ...]
    rho: tuple[float, ...]
    bin_size: int | None = None

    def __post_init__(self) -> None:
        if not self.alpha or len(self.alpha) != len(self.rho):
            raise ValueError("TETHER coefficient tables must be non-empty and aligned")
        if self.bin_size is None and len(self.alpha) != 1:
            raise ValueError("global TETHER needs exactly one coefficient pair")
        if self.bin_size is not None and (self.bin_size < 1 or len(self.alpha) < 2):
            raise ValueError("positioned TETHER needs at least two positive-width bins")
        if not all(math.isfinite(value) for value in (*self.alpha, *self.rho)):
            raise ValueError("TETHER coefficients must be finite")

    @property
    def num_bins(self) -> int:
        return len(self.alpha)

    def bin_index(self, action_position: int) -> int:
        if action_position < 0:
            raise ValueError("TETHER action position must be non-negative")
        if self.bin_size is None:
            return 0
        return min(action_position // self.bin_size, self.num_bins - 1)


@dataclass(frozen=True)
class TetherRegressionStats:
    """Token-weighted sufficient statistics for one coefficient bin."""

    weight: int = 0
    alpha_alpha: float = 0.0
    alpha_rho: float = 0.0
    rho_rho: float = 0.0
    alpha_target: float = 0.0
    rho_target: float = 0.0
    target_target: float = 0.0
    clipped: int = 0

    def __post_init__(self) -> None:
        if self.weight < 0 or self.clipped < 0 or self.clipped > self.weight:
            raise ValueError("invalid adaptive TETHER regression counts")
        moments = (
            self.alpha_alpha,
            self.alpha_rho,
            self.rho_rho,
            self.alpha_target,
            self.rho_target,
            self.target_target,
        )
        if not all(math.isfinite(value) for value in moments):
            raise ValueError("adaptive TETHER regression moments must be finite")
        if self.alpha_alpha < 0 or self.rho_rho < 0 or self.target_target < 0:
            raise ValueError("adaptive TETHER diagonal moments must be non-negative")

    def __add__(self, other: TetherRegressionStats) -> TetherRegressionStats:
        return TetherRegressionStats(
            weight=self.weight + other.weight,
            alpha_alpha=self.alpha_alpha + other.alpha_alpha,
            alpha_rho=self.alpha_rho + other.alpha_rho,
            rho_rho=self.rho_rho + other.rho_rho,
            alpha_target=self.alpha_target + other.alpha_target,
            rho_target=self.rho_target + other.rho_target,
            target_target=self.target_target + other.target_target,
            clipped=self.clipped + other.clipped,
        )


@dataclass(frozen=True)
class TetherRolloutStats:
    """One rollout's sufficient statistics, partitioned by position bin."""

    bins: tuple[TetherRegressionStats, ...]

    def __post_init__(self) -> None:
        if not self.bins:
            raise ValueError("adaptive TETHER rollout stats need at least one bin")


@dataclass
class _StatsAccumulator:
    weight: int = 0
    alpha_alpha: float = 0.0
    alpha_rho: float = 0.0
    rho_rho: float = 0.0
    alpha_target: float = 0.0
    rho_target: float = 0.0
    target_target: float = 0.0
    clipped: int = 0

    def add_row(self, alpha_feature: float, rho_feature: float, target: float, *, clipped: bool) -> None:
        self.weight += 1
        self.alpha_alpha += alpha_feature * alpha_feature
        self.alpha_rho += alpha_feature * rho_feature
        self.rho_rho += rho_feature * rho_feature
        self.alpha_target += alpha_feature * target
        self.rho_target += rho_feature * target
        self.target_target += target * target
        self.clipped += int(clipped)

    def freeze(self) -> TetherRegressionStats:
        return TetherRegressionStats(**asdict(self))


def tether_branch_advantages(
    *,
    reward: float,
    group_anchor: float,
    values: list[float],
    train_mask: list[bool],
    action_mask: list[bool],
    coefficients: TetherCoefficientTable,
    reward_range: tuple[float, float],
    collect_stats: bool = True,
) -> tuple[list[float], tuple[TetherRegressionStats, ...] | None]:
    """Score one branch and collect the adaptive moments in the same token pass.

    ``action_mask`` is native branch provenance: it chooses the causal branch
    start and advances position. ``train_mask`` is the deduplicated policy mask:
    it alone decides which shared action tokens receive a gradient/regression row.
    """

    if len(values) != len(train_mask) or len(values) != len(action_mask):
        raise ValueError("TETHER value, train-mask, and action-mask streams must align")
    if any(trainable and not sampled for trainable, sampled in zip(train_mask, action_mask, strict=True)):
        raise ValueError("TETHER train mask must be a subset of native sampled provenance")
    low, high = reward_range
    scale = high - low
    if scale <= 0 or not all(math.isfinite(value) for value in (reward, group_anchor, scale)):
        raise ValueError("TETHER received an invalid scalar or reward range")

    first_action = next((index for index, sampled in enumerate(action_mask) if sampled), None)
    output = [0.0] * len(values)
    accumulators = [_StatsAccumulator() for _ in range(coefficients.num_bins)] if collect_stats else None
    if first_action is None:
        stats = tuple(accumulator.freeze() for accumulator in accumulators) if accumulators is not None else None
        return output, stats

    start_value = values[first_action]
    if not math.isfinite(start_value):
        raise ValueError("TETHER received a non-finite start value")
    target = (reward - group_anchor) / scale
    alpha_feature = (start_value - group_anchor) / scale
    action_position = 0
    bin_size = coefficients.bin_size
    final_bin = coefficients.num_bins - 1
    for index, sampled in enumerate(action_mask):
        if not sampled:
            continue
        bin_index = 0 if bin_size is None else min(action_position // bin_size, final_bin)
        action_position += 1
        if not train_mask[index]:
            continue
        value = values[index]
        if not math.isfinite(value):
            raise ValueError("TETHER received a non-finite value prediction")
        alpha = coefficients.alpha[bin_index]
        rho = coefficients.rho[bin_index]
        raw_baseline = group_anchor + alpha * (start_value - group_anchor) + rho * (value - start_value)
        baseline = min(max(raw_baseline, low), high)
        output[index] = reward - baseline
        if accumulators is not None:
            accumulators[bin_index].add_row(
                alpha_feature,
                (value - start_value) / scale,
                target,
                clipped=raw_baseline != baseline,
            )
    stats = tuple(accumulator.freeze() for accumulator in accumulators) if accumulators is not None else None
    return output, stats


class AdaptiveTetherCoefficients:
    """Exact rollout-window ridge fits with deliberately uncorrected coefficient EMAs.

    Position bins fit independently. Unsupported bins hold their state; sparse
    supported bins receive stronger ridge and a smaller EMA update.
    """

    def __init__(
        self,
        config: AdaptiveTetherConfig,
        *,
        batch_size: int,
        reward_range: tuple[float, float] = (0.0, 1.0),
        num_bins: int = 1,
        bin_size: int | None = None,
        position_horizon: int | None = None,
    ):
        if batch_size < 1:
            raise ValueError("adaptive TETHER batch_size must be positive")
        if num_bins < 1 or num_bins > MAX_TETHER_POSITION_BINS:
            raise ValueError(f"adaptive TETHER num_bins must be in [1, {MAX_TETHER_POSITION_BINS}]")
        if (bin_size is None) != (num_bins == 1):
            raise ValueError("adaptive TETHER bin_size must be set exactly when position bins are active")
        low, high = reward_range
        if not math.isfinite(low) or not math.isfinite(high) or high <= low or not math.isfinite(high - low):
            raise ValueError("adaptive TETHER reward_range must be finite and increasing")

        default_min_rollouts = 1 if bin_size is None else min(batch_size, max(1, math.ceil(batch_size / 8)))
        min_bin_rollouts = config.min_bin_rollouts or default_min_rollouts
        if min_bin_rollouts > batch_size:
            raise ValueError("adaptive TETHER min_bin_rollouts cannot exceed batch_size")

        self.config = config
        self.batch_size = batch_size
        self.reward_range = reward_range
        self.num_bins = num_bins
        self.bin_size = bin_size
        self.position_horizon = position_horizon
        self.min_bin_rollouts = min_bin_rollouts
        self._alpha = [config.initial_alpha] * num_bins
        self._rho = [config.initial_rho] * num_bins
        self.pending_rollouts = 0
        self.pending_bins = [TetherRegressionStats() for _ in range(num_bins)]
        self.pending_contributors = [0] * num_bins
        self.updates = 0
        self.skipped_updates = 0
        self.bin_updates = [0] * num_bins
        self.bin_skipped_updates = [0] * num_bins
        self.bin_age = [0] * num_bins
        self.last_fit_alpha = list(self._alpha)
        self.last_fit_rho = list(self._rho)
        self.last_fit_valid = [False] * num_bins
        self.last_condition = [0.0] * num_bins
        self.last_contributors = [0] * num_bins
        self.last_tokens = [0] * num_bins
        self.last_mse_loo = 0.0
        self.last_mse_fit = 0.0
        self.last_mse_ema = 0.0
        self.last_fit_token_fraction = 0.0
        self.last_clip_fraction = 0.0

    @property
    def coefficient_table(self) -> TetherCoefficientTable:
        return TetherCoefficientTable(tuple(self._alpha), tuple(self._rho), self.bin_size)

    @property
    def coefficients(self) -> tuple[float, float]:
        if self.num_bins != 1:
            raise AttributeError("positioned TETHER has a coefficient table, not one coefficient pair")
        return self._alpha[0], self._rho[0]

    @property
    def alpha(self) -> float:
        return self.coefficients[0]

    @property
    def rho(self) -> float:
        return self.coefficients[1]

    def observe_group(self, stats: list[TetherRolloutStats]) -> None:
        """Queue a fully scored group, consuming exact rollout windows."""
        for rollout in stats:
            if len(rollout.bins) != self.num_bins:
                raise ValueError("adaptive TETHER rollout bin count mismatch")
            for index, bin_stats in enumerate(rollout.bins):
                self.pending_bins[index] += bin_stats
                self.pending_contributors[index] += int(bin_stats.weight > 0)
            self.pending_rollouts += 1
            if self.pending_rollouts == self.batch_size:
                self._update(self.pending_bins, self.pending_contributors)
                self.pending_rollouts = 0
                self.pending_bins = [TetherRegressionStats() for _ in range(self.num_bins)]
                self.pending_contributors = [0] * self.num_bins

    @staticmethod
    def _residual_sum(stats: TetherRegressionStats, alpha: float, rho: float) -> float:
        return max(
            stats.target_target
            - 2.0 * alpha * stats.alpha_target
            - 2.0 * rho * stats.rho_target
            + alpha * alpha * stats.alpha_alpha
            + 2.0 * alpha * rho * stats.alpha_rho
            + rho * rho * stats.rho_rho,
            0.0,
        )

    def _solve(self, stats: TetherRegressionStats, ridge_scale: float) -> tuple[float, float, float] | None:
        inv_weight = 1.0 / stats.weight
        ridge = self.config.ridge * ridge_scale
        aa = stats.alpha_alpha * inv_weight + ridge
        ar = stats.alpha_rho * inv_weight
        rr = stats.rho_rho * inv_weight + ridge
        ay = stats.alpha_target * inv_weight
        ry = stats.rho_target * inv_weight

        alpha_scale = math.sqrt(aa) if aa > 0 else 0.0
        rho_scale = math.sqrt(rr) if rr > 0 else 0.0
        correlation = ar / alpha_scale / rho_scale if alpha_scale > 0 and rho_scale > 0 else float("nan")
        determinant = 1.0 - correlation * correlation
        if not all(
            math.isfinite(value) for value in (alpha_scale, rho_scale, correlation, determinant)
        ) or determinant <= 8.0 * math.ulp(1.0):
            return None

        scaled_alpha_target = ay / alpha_scale
        scaled_rho_target = ry / rho_scale
        fit_alpha = (scaled_alpha_target - correlation * scaled_rho_target) / determinant / alpha_scale
        fit_rho = (scaled_rho_target - correlation * scaled_alpha_target) / determinant / rho_scale
        if not math.isfinite(fit_alpha) or not math.isfinite(fit_rho):
            return None

        matrix_scale = max(aa, rr, abs(ar))
        scaled_aa, scaled_ar, scaled_rr = aa / matrix_scale, ar / matrix_scale, rr / matrix_scale
        trace = scaled_aa + scaled_rr
        max_eigenvalue = 0.5 * (trace + math.hypot(scaled_aa - scaled_rr, 2.0 * scaled_ar))
        scaled_determinant = scaled_aa * scaled_rr - scaled_ar * scaled_ar
        min_eigenvalue = scaled_determinant / max_eigenvalue if max_eigenvalue > 0 else 0.0
        condition = max_eigenvalue / min_eigenvalue if min_eigenvalue > 0 else float("inf")
        return fit_alpha, fit_rho, condition

    def _update(self, stats: list[TetherRegressionStats], contributors: list[int]) -> None:
        total_weight = sum(item.weight for item in stats)
        if total_weight == 0:
            self.skipped_updates += 1
            self.last_fit_alpha = list(self._alpha)
            self.last_fit_rho = list(self._rho)
            self.last_mse_loo = self.last_mse_fit = self.last_mse_ema = 0.0
            self.last_fit_token_fraction = self.last_clip_fraction = 0.0
            self.last_fit_valid = [False] * self.num_bins
            self.last_condition = [float("inf")] * self.num_bins
            self.last_contributors = list(contributors)
            self.last_tokens = [item.weight for item in stats]
            self.bin_age = [age + 1 for age in self.bin_age]
            self.bin_skipped_updates = [count + 1 for count in self.bin_skipped_updates]
            get_logger().warning("Skipping adaptive TETHER window with no trainable tokens")
            return

        any_fit = False
        self.last_contributors = list(contributors)
        self.last_tokens = [item.weight for item in stats]
        self.last_fit_valid = [False] * self.num_bins
        for index, (bin_stats, support) in enumerate(zip(stats, contributors, strict=True)):
            self.bin_age[index] += 1
            self.last_fit_alpha[index] = self._alpha[index]
            self.last_fit_rho[index] = self._rho[index]
            self.last_condition[index] = float("inf")
            if bin_stats.weight == 0 or support < self.min_bin_rollouts:
                self.bin_skipped_updates[index] += 1
                continue
            # Normalizing within the bin keeps its fit independent of unrelated
            # sequence length; rollout support controls sparse-bin shrinkage.
            ridge_scale = self.batch_size / support if self.num_bins > 1 else 1.0
            solved = self._solve(bin_stats, ridge_scale)
            if solved is None:
                self.bin_skipped_updates[index] += 1
                continue
            fit_alpha, fit_rho, condition = solved
            decay = self.config.ema_decay
            if self.num_bins > 1:
                decay **= support / self.batch_size
            self._alpha[index] = decay * self._alpha[index] + (1.0 - decay) * fit_alpha
            self._rho[index] = decay * self._rho[index] + (1.0 - decay) * fit_rho
            self.last_fit_alpha[index] = fit_alpha
            self.last_fit_rho[index] = fit_rho
            self.last_fit_valid[index] = True
            self.last_condition[index] = condition
            self.bin_updates[index] += 1
            self.bin_age[index] = 0
            any_fit = True

        if any_fit:
            self.updates += 1
        else:
            self.skipped_updates += 1
        self.last_mse_loo = sum(item.target_target for item in stats) / total_weight
        fit_weight = sum(item.weight for item, valid in zip(stats, self.last_fit_valid, strict=True) if valid)
        self.last_mse_fit = (
            sum(
                self._residual_sum(item, alpha, rho)
                for item, alpha, rho, valid in zip(
                    stats,
                    self.last_fit_alpha,
                    self.last_fit_rho,
                    self.last_fit_valid,
                    strict=True,
                )
                if valid
            )
            / fit_weight
            if fit_weight
            else 0.0
        )
        self.last_mse_ema = (
            sum(
                self._residual_sum(item, alpha, rho)
                for item, alpha, rho in zip(stats, self._alpha, self._rho, strict=True)
            )
            / total_weight
        )
        self.last_fit_token_fraction = fit_weight / total_weight
        self.last_clip_fraction = sum(item.clipped for item in stats) / total_weight

        if self.num_bins == 1:
            get_logger().info(
                "Adaptive TETHER fit "
                f"{self.updates} | alpha={self._alpha[0]:.4f}, rho={self._rho[0]:.4f} | "
                f"batch fit=({self.last_fit_alpha[0]:.4f}, {self.last_fit_rho[0]:.4f}) | "
                f"mse LOO={self.last_mse_loo:.6g}, fit={self.last_mse_fit:.6g}"
            )
        else:
            get_logger().info(
                f"Adaptive positioned TETHER window | updated={sum(self.last_fit_valid)}/{self.num_bins} bins | "
                f"alpha=[{min(self._alpha):.3f}, {max(self._alpha):.3f}] | "
                f"rho=[{min(self._rho):.3f}, {max(self._rho):.3f}]"
            )

    def metrics(self) -> dict[str, float]:
        base = {
            "tether/updates": float(self.updates),
            "tether/skipped_updates": float(self.skipped_updates),
            "tether/pending_rollouts": float(self.pending_rollouts),
            "tether/regression_batch_size": float(self.batch_size),
            "tether/mse_loo": self.last_mse_loo,
            "tether/mse_batch_fit": self.last_mse_fit,
            "tether/mse_ema": self.last_mse_ema,
            "tether/fit_token_fraction": self.last_fit_token_fraction,
            "tether/clip_fraction": self.last_clip_fraction,
        }
        if self.num_bins == 1:
            base |= {
                "tether/alpha": self._alpha[0],
                "tether/rho": self._rho[0],
                "tether/batch_fit_alpha": self.last_fit_alpha[0],
                "tether/batch_fit_rho": self.last_fit_rho[0],
                "tether/batch_fit_valid": float(self.last_fit_valid[0]),
            }
            if math.isfinite(self.last_condition[0]):
                base["tether/condition_number"] = self.last_condition[0]
            return base

        active = sum(update > 0 for update in self.bin_updates)
        base |= {
            "tether/position/num_bins": float(self.num_bins),
            "tether/position/active_bins": float(active),
            "tether/position/min_bin_rollouts": float(self.min_bin_rollouts),
            "tether/position/alpha_min": min(self._alpha),
            "tether/position/alpha_max": max(self._alpha),
            "tether/position/rho_min": min(self._rho),
            "tether/position/rho_max": max(self._rho),
        }
        finite_conditions = [value for value in self.last_condition if math.isfinite(value)]
        if finite_conditions:
            base["tether/condition_number"] = max(finite_conditions)
        for index in range(self.num_bins):
            prefix = f"tether/position/bin_{index:03d}"
            base |= {
                f"{prefix}/alpha": self._alpha[index],
                f"{prefix}/rho": self._rho[index],
                f"{prefix}/batch_fit_alpha": self.last_fit_alpha[index],
                f"{prefix}/batch_fit_rho": self.last_fit_rho[index],
                f"{prefix}/fit_valid": float(self.last_fit_valid[index]),
                f"{prefix}/contributing_rollouts": float(self.last_contributors[index]),
                f"{prefix}/tokens": float(self.last_tokens[index]),
                f"{prefix}/updates": float(self.bin_updates[index]),
                f"{prefix}/skipped_updates": float(self.bin_skipped_updates[index]),
                f"{prefix}/age": float(self.bin_age[index]),
            }
            if math.isfinite(self.last_condition[index]):
                base[f"{prefix}/condition_number"] = self.last_condition[index]
        return base

    def metric_keys(self) -> list[str]:
        keys = list(self.metrics())
        if "tether/condition_number" not in keys:
            keys.append("tether/condition_number")
        if self.num_bins > 1:
            for index in range(self.num_bins):
                key = f"tether/position/bin_{index:03d}/condition_number"
                if key not in keys:
                    keys.append(key)
        return keys

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "batch_size": self.batch_size,
            "reward_range": self.reward_range,
            "num_bins": self.num_bins,
            "bin_size": self.bin_size,
            "position_horizon": self.position_horizon,
            "min_bin_rollouts": self.min_bin_rollouts,
            "alpha": list(self._alpha),
            "rho": list(self._rho),
            "pending_rollouts": self.pending_rollouts,
            "pending_bins": [asdict(stats) for stats in self.pending_bins],
            "pending_contributors": list(self.pending_contributors),
            "updates": self.updates,
            "skipped_updates": self.skipped_updates,
            "bin_updates": list(self.bin_updates),
            "bin_skipped_updates": list(self.bin_skipped_updates),
            "bin_age": list(self.bin_age),
            "last_fit_alpha": list(self.last_fit_alpha),
            "last_fit_rho": list(self.last_fit_rho),
            "last_fit_valid": list(self.last_fit_valid),
            "last_condition": list(self.last_condition),
            "last_contributors": list(self.last_contributors),
            "last_tokens": list(self.last_tokens),
            "last_mse_loo": self.last_mse_loo,
            "last_mse_fit": self.last_mse_fit,
            "last_mse_ema": self.last_mse_ema,
            "last_fit_token_fraction": self.last_fit_token_fraction,
            "last_clip_fraction": self.last_clip_fraction,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        schema_version = self._int_scalar(state.get("schema_version", 1), "schema version", minimum=1)
        if schema_version == 1:
            self._load_scalar_state(state)
            return
        if schema_version != 2:
            raise ValueError(f"unsupported adaptive TETHER checkpoint schema {schema_version}")
        expected = (
            self.batch_size,
            self.reward_range,
            self.num_bins,
            self.bin_size,
            self.position_horizon,
            self.min_bin_rollouts,
        )
        saved_bin_size = state.get("bin_size")
        if saved_bin_size is not None:
            saved_bin_size = self._int_scalar(saved_bin_size, "bin size", minimum=1)
        saved_horizon = state.get("position_horizon")
        if saved_horizon is not None:
            saved_horizon = self._int_scalar(saved_horizon, "position horizon", minimum=1)
        saved = (
            self._int_scalar(state["batch_size"], "batch size", minimum=1),
            tuple(state["reward_range"]),
            self._int_scalar(state["num_bins"], "num bins", minimum=1),
            saved_bin_size,
            saved_horizon,
            self._int_scalar(state["min_bin_rollouts"], "minimum bin rollouts", minimum=1),
        )
        if saved != expected:
            raise ValueError(f"adaptive TETHER checkpoint contract {saved} does not match config {expected}")

        self._alpha = self._float_vector(state["alpha"], "alpha")
        self._rho = self._float_vector(state["rho"], "rho")
        self.pending_rollouts = self._int_scalar(state.get("pending_rollouts", 0), "pending rollouts")
        self.pending_bins = [TetherRegressionStats(**item) for item in state.get("pending_bins", [])]
        self.pending_contributors = self._int_vector(state.get("pending_contributors", []), "pending contributors")
        self.updates = self._int_scalar(state.get("updates", 0), "updates")
        self.skipped_updates = self._int_scalar(state.get("skipped_updates", 0), "skipped updates")
        self.bin_updates = self._int_vector(state.get("bin_updates", []), "bin updates")
        self.bin_skipped_updates = self._int_vector(state.get("bin_skipped_updates", []), "bin skipped updates")
        self.bin_age = self._int_vector(state.get("bin_age", []), "bin age")
        self.last_fit_alpha = self._float_vector(state.get("last_fit_alpha", self._alpha), "last fit alpha")
        self.last_fit_rho = self._float_vector(state.get("last_fit_rho", self._rho), "last fit rho")
        self.last_fit_valid = self._bool_vector(state.get("last_fit_valid", []), "last fit valid")
        self.last_condition = self._condition_vector(state.get("last_condition", []))
        self.last_contributors = self._int_vector(state.get("last_contributors", []), "last contributors")
        self.last_tokens = self._int_vector(state.get("last_tokens", []), "last tokens")
        self.last_mse_loo = float(state.get("last_mse_loo", 0.0))
        self.last_mse_fit = float(state.get("last_mse_fit", 0.0))
        self.last_mse_ema = float(state.get("last_mse_ema", 0.0))
        self.last_fit_token_fraction = float(state.get("last_fit_token_fraction", 0.0))
        self.last_clip_fraction = float(state.get("last_clip_fraction", 0.0))
        self._validate_loaded_state()

    def _load_scalar_state(self, state: dict[str, Any]) -> None:
        if self.num_bins != 1:
            raise ValueError("scalar adaptive TETHER checkpoints cannot initialize position bins")
        if self._int_scalar(state["batch_size"], "batch size", minimum=1) != self.batch_size:
            raise ValueError("adaptive TETHER checkpoint batch_size does not match config")
        if tuple(state.get("reward_range", self.reward_range)) != self.reward_range:
            raise ValueError("adaptive TETHER checkpoint reward_range does not match config")
        self._alpha = [float(state["alpha"])]
        self._rho = [float(state["rho"])]
        pending = [TetherRegressionStats(**item) for item in state.get("pending", [])]
        self.pending_rollouts = len(pending)
        pending_total = TetherRegressionStats()
        for item in pending:
            pending_total += item
        self.pending_bins = [pending_total]
        self.pending_contributors = [sum(item.weight > 0 for item in pending)]
        self.updates = self._int_scalar(state.get("updates", 0), "updates")
        self.skipped_updates = self._int_scalar(state.get("skipped_updates", 0), "skipped updates")
        self.bin_updates = [self.updates]
        self.bin_skipped_updates = [self.skipped_updates]
        self.bin_age = [0]
        self.last_fit_alpha = [float(state.get("last_fit_alpha", self._alpha[0]))]
        self.last_fit_rho = [float(state.get("last_fit_rho", self._rho[0]))]
        fit_valid = state.get("last_fit_valid", self.updates > 0)
        if not isinstance(fit_valid, bool):
            raise ValueError("adaptive TETHER checkpoint contains invalid last fit valid")
        self.last_fit_valid = [fit_valid]
        self.last_condition = [float(state.get("last_condition", 0.0))]
        self.last_contributors = [0]
        self.last_tokens = [0]
        self.last_mse_loo = float(state.get("last_mse_loo", 0.0))
        self.last_mse_fit = float(state.get("last_mse_fit", 0.0))
        self.last_mse_ema = float(state.get("last_mse_ema", 0.0))
        self.last_fit_token_fraction = float(state.get("last_fit_token_fraction", float(self.last_fit_valid[0])))
        self.last_clip_fraction = float(state.get("last_clip_fraction", 0.0))
        self._validate_loaded_state()

    def _float_vector(self, values: list[Any], name: str) -> list[float]:
        result = [float(value) for value in values]
        if len(result) != self.num_bins or not all(math.isfinite(value) for value in result):
            raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
        return result

    @staticmethod
    def _int_scalar(value: Any, name: str, *, minimum: int = 0) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
        return value

    def _bool_vector(self, values: list[Any], name: str) -> list[bool]:
        if len(values) != self.num_bins or any(not isinstance(value, bool) for value in values):
            raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
        return list(values)

    def _int_vector(self, values: list[Any], name: str) -> list[int]:
        result = [int(value) for value in values]
        if (
            len(result) != self.num_bins
            or any(isinstance(value, bool) or parsed != value for value, parsed in zip(values, result, strict=True))
            or any(value < 0 for value in result)
        ):
            raise ValueError(f"adaptive TETHER checkpoint contains invalid {name}")
        return result

    def _condition_vector(self, values: list[Any]) -> list[float]:
        result = [float(value) for value in values]
        if len(result) != self.num_bins or any(math.isnan(value) or value < 0 for value in result):
            raise ValueError("adaptive TETHER checkpoint contains invalid condition numbers")
        return result

    def _validate_loaded_state(self) -> None:
        vectors = (
            self.pending_bins,
            self.last_fit_valid,
            self.last_condition,
            self.last_contributors,
            self.last_tokens,
        )
        if any(len(values) != self.num_bins for values in vectors):
            raise ValueError("adaptive TETHER checkpoint vector lengths do not match num_bins")
        if not 0 <= self.pending_rollouts < self.batch_size:
            raise ValueError("adaptive TETHER checkpoint contains invalid pending rollout count")
        if any(value > self.pending_rollouts for value in self.pending_contributors):
            raise ValueError("adaptive TETHER checkpoint contains invalid pending bin support")
        if self.updates < 0 or self.skipped_updates < 0:
            raise ValueError("adaptive TETHER checkpoint contains invalid update counts")
        finite_state = (
            *self._alpha,
            *self._rho,
            *self.last_fit_alpha,
            *self.last_fit_rho,
            self.last_mse_loo,
            self.last_mse_fit,
            self.last_mse_ema,
            self.last_fit_token_fraction,
            self.last_clip_fraction,
        )
        if not all(math.isfinite(value) for value in finite_state):
            raise ValueError("adaptive TETHER checkpoint contains non-finite diagnostics")
        if (
            any(value < 0 for value in (self.last_mse_loo, self.last_mse_fit, self.last_mse_ema))
            or not 0.0 <= self.last_fit_token_fraction <= 1.0
            or not 0.0 <= self.last_clip_fraction <= 1.0
        ):
            raise ValueError("adaptive TETHER checkpoint contains invalid diagnostics")


class TetherRuntime:
    """Compile and score one TETHER baseline without leaking it into GRPO plumbing."""

    def __init__(
        self,
        config: TetherBaselineConfig,
        *,
        value_seq_len: int,
        policy_seq_len: int,
        adaptive_batch_size: int | None,
    ) -> None:
        self.config = config
        self.value_seq_len = value_seq_len
        self.policy_seq_len = policy_seq_len

        position = config.position
        if position is None:
            self.position_horizon = None
            self.bin_size = None
            self.num_bins = 1
        else:
            horizon = position.max_action_tokens or policy_seq_len
            if horizon > policy_seq_len:
                raise ValueError("TETHER position max_action_tokens cannot exceed policy sequence length")
            self.position_horizon = horizon
            self.bin_size = position.bin_size
            self.num_bins = math.ceil(horizon / position.bin_size)
            if self.num_bins < 2:
                raise ValueError("TETHER position conditioning needs at least two bins")
            if self.num_bins > MAX_TETHER_POSITION_BINS:
                raise ValueError(
                    f"TETHER position config creates {self.num_bins} bins; maximum is {MAX_TETHER_POSITION_BINS}"
                )

        self.adaptive: AdaptiveTetherCoefficients | None = None
        if config.adaptive is not None:
            if adaptive_batch_size is None:
                raise ValueError("adaptive TETHER needs a resolved rollout batch size")
            self.adaptive = AdaptiveTetherCoefficients(
                config.adaptive,
                batch_size=adaptive_batch_size,
                reward_range=config.reward_range,
                num_bins=self.num_bins,
                bin_size=self.bin_size,
                position_horizon=self.position_horizon,
            )

        if self.num_bins == 1:
            self.static_coefficients = TetherCoefficientTable((config.alpha,), (config.rho,), self.bin_size)
        else:
            denominator = self.num_bins - 1
            self.static_coefficients = TetherCoefficientTable(
                tuple(config.alpha * index / denominator for index in range(self.num_bins)),
                tuple(config.rho * index / denominator for index in range(self.num_bins)),
                self.bin_size,
            )

    @property
    def coefficients(self) -> TetherCoefficientTable:
        return self.adaptive.coefficient_table if self.adaptive is not None else self.static_coefficients

    def score_group(self, group: list[Rollout]) -> None:
        anchors = group_baselines([float(rollout.reward) for rollout in group], self.config.group)
        coefficient_snapshot = self.coefficients
        regression_stats: list[TetherRolloutStats] = []
        for rollout, group_anchor in zip(group, anchors, strict=True):
            if rollout.value_predictions is None:
                raise RuntimeError("value evaluator did not attach TETHER predictions")
            branch_views = list(iter_trainable_branches(rollout))
            if len(branch_views) != len(rollout.samples):
                raise ValueError("TETHER branch/sample alignment mismatch")
            corrected: list[float] = []
            rollout_bins = (
                [TetherRegressionStats() for _ in range(self.num_bins)] if self.adaptive is not None else None
            )
            for (branch, train_mask), sample, values in zip(
                branch_views,
                rollout.samples,
                rollout.value_predictions,
                strict=True,
            ):
                if sample.token_ids != branch.token_ids or sample.mask != train_mask:
                    raise ValueError("TETHER branch/sample token streams are misaligned")
                value_length = min(len(sample.token_ids), self.value_seq_len, self.policy_seq_len)
                branch_advantages, branch_stats = tether_branch_advantages(
                    reward=float(rollout.reward),
                    group_anchor=group_anchor,
                    values=values[:value_length],
                    train_mask=train_mask[:value_length],
                    action_mask=branch.sampled_mask[:value_length],
                    coefficients=coefficient_snapshot,
                    reward_range=self.config.reward_range,
                    collect_stats=self.adaptive is not None,
                )
                corrected.extend(branch_advantages + [0.0] * (len(sample.token_ids) - value_length))
                if rollout_bins is not None:
                    assert branch_stats is not None
                    rollout_bins = [
                        current + incoming for current, incoming in zip(rollout_bins, branch_stats, strict=True)
                    ]
            rollout.assign_advantages(corrected)
            if rollout_bins is not None:
                regression_stats.append(TetherRolloutStats(tuple(rollout_bins)))

        if self.adaptive is not None:
            self.adaptive.observe_group(regression_stats)

    def metrics(self) -> dict[str, float]:
        return self.adaptive.metrics() if self.adaptive is not None else {}

    def metric_keys(self) -> list[str]:
        return self.adaptive.metric_keys() if self.adaptive is not None else []

    def state_dict(self) -> dict[str, Any]:
        return self.adaptive.state_dict() if self.adaptive is not None else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.adaptive is None:
            raise ValueError("checkpoint contains adaptive TETHER state but adaptive mode is disabled")
        self.adaptive.load_state_dict(state)
