"""Adaptive coefficient fitting for the TETHER baseline.

The actor still uses the ordinary TETHER formula. This module only estimates
its two global control-variate coefficients from completed, earlier rollouts.
It stores one small sufficient-statistics record per pending rollout; raw
tokens and value predictions are never retained by the estimator.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from prime_rl.configs.algorithm import AdaptiveTetherConfig
from prime_rl.utils.logger import get_logger


@dataclass(frozen=True)
class TetherRegressionStats:
    """Token-weighted sufficient statistics for one rollout.

    Features and target are divided by the configured reward-range width. This
    leaves the fitted coefficients unchanged while making ``ridge`` invariant
    to a linear rescaling of rewards and values.
    """

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


def tether_regression_stats(
    *,
    reward: float,
    group_anchor: float,
    value_predictions: list[list[float]],
    masks: list[list[bool]],
    reward_range: tuple[float, float],
    alpha: float,
    rho: float,
) -> TetherRegressionStats:
    """Build one rollout's moments for ``y ~ alpha*x_alpha + rho*x_rho``.

    ``y = R - B_loo``, ``x_alpha = V_0 - B_loo``, and
    ``x_rho = V_t - V_0``. Each trainable token is one regression row, matching
    the actor loss's token normalization. A rollout branch gets its own causal
    start value ``V_0``.
    """

    if len(value_predictions) != len(masks):
        raise ValueError("value prediction/mask branch count mismatch")
    low, high = reward_range
    scale = high - low
    if scale <= 0 or not all(math.isfinite(value) for value in (reward, group_anchor, scale, alpha, rho)):
        raise ValueError("adaptive tether received an invalid scalar or reward range")

    weight = 0
    alpha_alpha = alpha_rho = rho_rho = 0.0
    alpha_target = rho_target = target_target = 0.0
    clipped = 0
    target = (reward - group_anchor) / scale
    for values, mask in zip(value_predictions, masks, strict=True):
        if len(values) != len(mask):
            raise ValueError("value prediction/mask length mismatch")
        first_action = next((index for index, trainable in enumerate(mask) if trainable), None)
        if first_action is None:
            continue
        start_value = values[first_action]
        if not math.isfinite(start_value):
            raise ValueError("adaptive tether received a non-finite value prediction")
        alpha_feature = (start_value - group_anchor) / scale
        for value, trainable in zip(values, mask, strict=True):
            if not trainable:
                continue
            if not math.isfinite(value):
                raise ValueError("adaptive tether received a non-finite value prediction")
            rho_feature = (value - start_value) / scale
            alpha_alpha += alpha_feature * alpha_feature
            alpha_rho += alpha_feature * rho_feature
            rho_rho += rho_feature * rho_feature
            alpha_target += alpha_feature * target
            rho_target += rho_feature * target
            target_target += target * target
            raw_baseline = group_anchor + alpha * (start_value - group_anchor) + rho * (value - start_value)
            clipped += int(raw_baseline < low or raw_baseline > high)
            weight += 1

    return TetherRegressionStats(
        weight=weight,
        alpha_alpha=alpha_alpha,
        alpha_rho=alpha_rho,
        rho_rho=rho_rho,
        alpha_target=alpha_target,
        rho_target=rho_target,
        target_target=target_target,
        clipped=clipped,
    )


class AdaptiveTetherCoefficients:
    """Exact rollout-batched ridge fits followed by a deliberately uncorrected EMA.

    The zero-initialized EMA is intentionally *not* bias-corrected: its early
    shrinkage is the safe ramp from LOO requested by the adaptive mode. The
    policy-gradient baseline remains action-independent because the owner only
    calls :meth:`observe_group` after scoring the complete group and applies an
    update to later groups.
    """

    def __init__(
        self,
        config: AdaptiveTetherConfig,
        *,
        batch_size: int,
        reward_range: tuple[float, float] = (0.0, 1.0),
    ):
        if batch_size < 1:
            raise ValueError("adaptive TETHER batch_size must be positive")
        low, high = reward_range
        if not math.isfinite(low) or not math.isfinite(high) or high <= low or not math.isfinite(high - low):
            raise ValueError("adaptive TETHER reward_range must be finite and increasing")
        self.config = config
        self.batch_size = batch_size
        self.reward_range = reward_range
        self.alpha = config.initial_alpha
        self.rho = config.initial_rho
        self.pending: deque[TetherRegressionStats] = deque()
        self.updates = 0
        self.skipped_updates = 0
        self.last_fit_alpha = config.initial_alpha
        self.last_fit_rho = config.initial_rho
        self.last_fit_valid = False
        self.last_condition = 0.0
        self.last_mse_loo = 0.0
        self.last_mse_fit = 0.0
        self.last_mse_ema = 0.0
        self.last_clip_fraction = 0.0

    @property
    def coefficients(self) -> tuple[float, float]:
        return self.alpha, self.rho

    def observe_group(self, stats: list[TetherRegressionStats]) -> None:
        """Queue a fully scored group, then consume any exact rollout batches."""
        self.pending.extend(stats)
        while len(self.pending) >= self.batch_size:
            batch = TetherRegressionStats()
            for _ in range(self.batch_size):
                batch += self.pending.popleft()
            self._update(batch)

    @staticmethod
    def _residual_mse(stats: TetherRegressionStats, alpha: float, rho: float) -> float:
        if stats.weight == 0:
            return 0.0
        squared_error = (
            stats.target_target
            - 2.0 * alpha * stats.alpha_target
            - 2.0 * rho * stats.rho_target
            + alpha * alpha * stats.alpha_alpha
            + 2.0 * alpha * rho * stats.alpha_rho
            + rho * rho * stats.rho_rho
        )
        return max(squared_error / stats.weight, 0.0)

    def _update(self, stats: TetherRegressionStats) -> None:
        if stats.weight == 0:
            self.skipped_updates += 1
            self.last_fit_alpha = self.alpha
            self.last_fit_rho = self.rho
            self.last_fit_valid = False
            self.last_condition = float("inf")
            self.last_mse_loo = 0.0
            self.last_mse_fit = 0.0
            self.last_mse_ema = 0.0
            self.last_clip_fraction = 0.0
            get_logger().warning("Skipping adaptive TETHER fit with no trainable tokens")
            return

        inv_weight = 1.0 / stats.weight
        aa = stats.alpha_alpha * inv_weight + self.config.ridge
        ar = stats.alpha_rho * inv_weight
        rr = stats.rho_rho * inv_weight + self.config.ridge
        ay = stats.alpha_target * inv_weight
        ry = stats.rho_target * inv_weight

        # Diagonal equilibration avoids overflowing ``aa * rr`` and remains
        # accurate when the alpha and rho features have very different scales.
        # The resulting correlation matrix is [[1, c], [c, 1]].
        alpha_scale = math.sqrt(aa) if aa > 0 else 0.0
        rho_scale = math.sqrt(rr) if rr > 0 else 0.0
        correlation = ar / alpha_scale / rho_scale if alpha_scale > 0 and rho_scale > 0 else float("nan")
        determinant = 1.0 - correlation * correlation
        singular_tolerance = 8.0 * math.ulp(1.0)
        if (
            not all(math.isfinite(value) for value in (alpha_scale, rho_scale, correlation, determinant))
            or determinant <= singular_tolerance
        ):
            self.skipped_updates += 1
            self._record_skipped_fit(stats)
            get_logger().warning("Skipping singular adaptive TETHER regression fit")
            return

        scaled_alpha_target = ay / alpha_scale
        scaled_rho_target = ry / rho_scale
        fit_alpha = (scaled_alpha_target - correlation * scaled_rho_target) / determinant / alpha_scale
        fit_rho = (scaled_rho_target - correlation * scaled_alpha_target) / determinant / rho_scale
        if not math.isfinite(fit_alpha) or not math.isfinite(fit_rho):
            self.skipped_updates += 1
            self._record_skipped_fit(stats)
            get_logger().warning("Skipping non-finite adaptive TETHER regression fit")
            return

        matrix_scale = max(aa, rr, abs(ar))
        scaled_aa, scaled_ar, scaled_rr = aa / matrix_scale, ar / matrix_scale, rr / matrix_scale
        trace = scaled_aa + scaled_rr
        max_eigenvalue = 0.5 * (trace + math.hypot(scaled_aa - scaled_rr, 2.0 * scaled_ar))
        scaled_determinant = scaled_aa * scaled_rr - scaled_ar * scaled_ar
        min_eigenvalue = scaled_determinant / max_eigenvalue if max_eigenvalue > 0 else 0.0
        condition = max_eigenvalue / min_eigenvalue if min_eigenvalue > 0 else float("inf")

        decay = self.config.ema_decay
        self.alpha = decay * self.alpha + (1.0 - decay) * fit_alpha
        self.rho = decay * self.rho + (1.0 - decay) * fit_rho
        self.last_fit_alpha = fit_alpha
        self.last_fit_rho = fit_rho
        self.last_fit_valid = True
        self.last_condition = condition
        self.last_mse_loo = stats.target_target * inv_weight
        self.last_mse_fit = self._residual_mse(stats, fit_alpha, fit_rho)
        self.last_mse_ema = self._residual_mse(stats, self.alpha, self.rho)
        self.last_clip_fraction = stats.clipped * inv_weight
        self.updates += 1
        get_logger().info(
            "Adaptive TETHER fit "
            f"{self.updates} | alpha={self.alpha:.4f}, rho={self.rho:.4f} | "
            f"batch fit=({fit_alpha:.4f}, {fit_rho:.4f}) | "
            f"mse LOO={self.last_mse_loo:.6g}, fit={self.last_mse_fit:.6g}"
        )

    def _record_skipped_fit(self, stats: TetherRegressionStats) -> None:
        """Replace stale batch diagnostics when a fit cannot be solved."""
        inv_weight = 1.0 / stats.weight
        current_mse = self._residual_mse(stats, self.alpha, self.rho)
        self.last_fit_alpha = self.alpha
        self.last_fit_rho = self.rho
        self.last_fit_valid = False
        self.last_condition = float("inf")
        self.last_mse_loo = stats.target_target * inv_weight
        self.last_mse_fit = current_mse
        self.last_mse_ema = current_mse
        self.last_clip_fraction = stats.clipped * inv_weight

    def metrics(self) -> dict[str, float]:
        metrics = {
            "tether/alpha": self.alpha,
            "tether/rho": self.rho,
            "tether/batch_fit_alpha": self.last_fit_alpha,
            "tether/batch_fit_rho": self.last_fit_rho,
            "tether/batch_fit_valid": float(self.last_fit_valid),
            "tether/updates": float(self.updates),
            "tether/skipped_updates": float(self.skipped_updates),
            "tether/pending_rollouts": float(len(self.pending)),
            "tether/regression_batch_size": float(self.batch_size),
            "tether/mse_loo": self.last_mse_loo,
            "tether/mse_batch_fit": self.last_mse_fit,
            "tether/mse_ema": self.last_mse_ema,
            "tether/clip_fraction": self.last_clip_fraction,
        }
        if math.isfinite(self.last_condition):
            metrics["tether/condition_number"] = self.last_condition
        return metrics

    def metric_keys(self) -> list[str]:
        keys = list(self.metrics())
        if "tether/condition_number" not in keys:
            keys.append("tether/condition_number")
        return keys

    def state_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "reward_range": self.reward_range,
            "alpha": self.alpha,
            "rho": self.rho,
            "pending": [asdict(stats) for stats in self.pending],
            "updates": self.updates,
            "skipped_updates": self.skipped_updates,
            "last_fit_alpha": self.last_fit_alpha,
            "last_fit_rho": self.last_fit_rho,
            "last_fit_valid": self.last_fit_valid,
            "last_condition": self.last_condition,
            "last_mse_loo": self.last_mse_loo,
            "last_mse_fit": self.last_mse_fit,
            "last_mse_ema": self.last_mse_ema,
            "last_clip_fraction": self.last_clip_fraction,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_batch_size = int(state["batch_size"])
        if saved_batch_size != self.batch_size:
            raise ValueError(
                f"adaptive TETHER checkpoint batch_size={saved_batch_size} does not match config {self.batch_size}"
            )
        saved_reward_range = tuple(state.get("reward_range", self.reward_range))
        if saved_reward_range != self.reward_range:
            raise ValueError(
                f"adaptive TETHER checkpoint reward_range={saved_reward_range} "
                f"does not match config {self.reward_range}"
            )
        self.alpha = float(state["alpha"])
        self.rho = float(state["rho"])
        self.pending = deque(TetherRegressionStats(**item) for item in state.get("pending", []))
        self.updates = int(state.get("updates", 0))
        self.skipped_updates = int(state.get("skipped_updates", 0))
        self.last_fit_alpha = float(state.get("last_fit_alpha", self.alpha))
        self.last_fit_rho = float(state.get("last_fit_rho", self.rho))
        self.last_fit_valid = bool(state.get("last_fit_valid", self.updates > 0))
        self.last_condition = float(state.get("last_condition", 0.0))
        self.last_mse_loo = float(state.get("last_mse_loo", 0.0))
        self.last_mse_fit = float(state.get("last_mse_fit", 0.0))
        self.last_mse_ema = float(state.get("last_mse_ema", 0.0))
        self.last_clip_fraction = float(state.get("last_clip_fraction", 0.0))
        finite = (
            self.alpha,
            self.rho,
            self.last_fit_alpha,
            self.last_fit_rho,
            self.last_mse_loo,
            self.last_mse_fit,
            self.last_mse_ema,
            self.last_clip_fraction,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("adaptive TETHER checkpoint contains non-finite state")
        if math.isnan(self.last_condition) or self.last_condition < 0:
            raise ValueError("adaptive TETHER checkpoint contains an invalid condition number")
        if self.updates < 0 or self.skipped_updates < 0 or len(self.pending) >= self.batch_size:
            raise ValueError("adaptive TETHER checkpoint contains invalid update counts")
        if (
            self.last_mse_loo < 0
            or self.last_mse_fit < 0
            or self.last_mse_ema < 0
            or not 0.0 <= self.last_clip_fraction <= 1.0
        ):
            raise ValueError("adaptive TETHER checkpoint contains invalid diagnostics")
