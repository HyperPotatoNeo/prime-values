from dataclasses import asdict

import pytest

from prime_rl.configs.algorithm import AdaptiveTetherConfig
from prime_rl.orchestrator.algo.tether import (
    AdaptiveTetherCoefficients,
    TetherRegressionStats,
    tether_regression_stats,
)


def _row(alpha_feature: float, rho_feature: float, target: float) -> TetherRegressionStats:
    return TetherRegressionStats(
        weight=1,
        alpha_alpha=alpha_feature**2,
        alpha_rho=alpha_feature * rho_feature,
        rho_rho=rho_feature**2,
        alpha_target=alpha_feature * target,
        rho_target=rho_feature * target,
        target_target=target**2,
    )


def test_joint_regression_recovers_two_factors_exactly():
    # X = [[1, 0], [0, 1], [1, 1]], beta = [2, -1].
    estimator = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0),
        batch_size=3,
    )
    estimator.observe_group([_row(1.0, 0.0, 2.0), _row(0.0, 1.0, -1.0), _row(1.0, 1.0, 1.0)])

    assert estimator.coefficients == pytest.approx((2.0, -1.0))
    assert estimator.last_mse_fit == pytest.approx(0.0, abs=1e-12)
    assert estimator.updates == 1


def test_zero_initialized_ema_is_intentionally_not_bias_corrected():
    estimator = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.9),
        batch_size=3,
    )
    estimator.observe_group([_row(1.0, 0.0, 2.0), _row(0.0, 1.0, -1.0), _row(1.0, 1.0, 1.0)])

    assert estimator.coefficients == pytest.approx((0.2, -0.1))


def test_rollout_moments_are_token_weighted_and_branch_local():
    stats = tether_regression_stats(
        reward=1.0,
        group_anchor=0.5,
        value_predictions=[[0.7, 1.0], [0.2, 0.4]],
        masks=[[True, True], [True, True]],
        reward_range=(0.0, 1.0),
        alpha=0.0,
        rho=0.0,
    )

    # B_loo=.5. Branch features are (.2, [0,.3]) and (-.3, [0,.2]); y=.5.
    assert stats.weight == 4
    assert stats.alpha_alpha == pytest.approx(2 * 0.2**2 + 2 * 0.3**2)
    assert stats.alpha_rho == pytest.approx(0.2 * 0.3 - 0.3 * 0.2)
    assert stats.rho_rho == pytest.approx(0.3**2 + 0.2**2)
    assert stats.alpha_target == pytest.approx(2 * 0.2 * 0.5 - 2 * 0.3 * 0.5)
    assert stats.rho_target == pytest.approx((0.3 + 0.2) * 0.5)
    assert stats.target_target == pytest.approx(4 * 0.5**2)


def test_exact_rollout_batches_retain_overflow():
    estimator = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=1e-6, ema_decay=0.0),
        batch_size=2,
    )
    rows = [_row(1.0, 0.0, 1.0), _row(-1.0, 0.0, -1.0), _row(0.0, 1.0, 2.0)]
    estimator.observe_group(rows)

    assert estimator.updates == 1
    assert len(estimator.pending) == 1
    estimator.observe_group([_row(0.0, -1.0, -2.0)])
    assert estimator.updates == 2
    assert len(estimator.pending) == 0


def test_normalized_ridge_is_invariant_to_duplicating_every_row():
    rows = [_row(1.0, 0.0, 2.0), _row(0.0, 1.0, -1.0), _row(1.0, 1.0, 1.0)]
    once = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.2, ema_decay=0.0),
        batch_size=3,
    )
    twice = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.2, ema_decay=0.0),
        batch_size=6,
    )
    once.observe_group(rows)
    twice.observe_group(rows + rows)

    assert twice.coefficients == pytest.approx(once.coefficients)


def test_reward_range_normalization_is_scale_invariant():
    unit = tether_regression_stats(
        reward=1.0,
        group_anchor=0.4,
        value_predictions=[[0.7, 0.9]],
        masks=[[True, True]],
        reward_range=(0.0, 1.0),
        alpha=0.2,
        rho=0.3,
    )
    scaled = tether_regression_stats(
        reward=10.0,
        group_anchor=4.0,
        value_predictions=[[7.0, 9.0]],
        masks=[[True, True]],
        reward_range=(0.0, 10.0),
        alpha=0.2,
        rho=0.3,
    )

    assert asdict(scaled) == pytest.approx(asdict(unit))


def test_singular_unregularized_batch_is_skipped_but_ridge_is_safe():
    no_ridge = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0),
        batch_size=1,
    )
    no_ridge.observe_group([_row(0.0, 0.0, 1.0)])
    assert no_ridge.updates == 0
    assert no_ridge.skipped_updates == 1
    assert no_ridge.last_condition == float("inf")
    assert "tether/condition_number" in no_ridge.metric_keys()
    assert "tether/condition_number" not in no_ridge.metrics()

    ridge = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=1e-6, ema_decay=0.0),
        batch_size=1,
    )
    ridge.observe_group([_row(0.0, 0.0, 1.0)])
    assert ridge.updates == 1
    assert ridge.coefficients == (0.0, 0.0)


def test_indefinite_moments_are_rejected_and_replace_stale_diagnostics():
    estimator = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0),
        batch_size=1,
    )
    estimator.observe_group(
        [
            TetherRegressionStats(
                weight=1,
                alpha_alpha=1.0,
                alpha_rho=2.0,
                rho_rho=1.0,
                alpha_target=1.0,
                rho_target=1.0,
                target_target=1.0,
            )
        ]
    )

    assert estimator.updates == 0
    assert estimator.skipped_updates == 1
    assert estimator.last_condition == float("inf")


def test_equilibrated_solve_handles_tiny_well_conditioned_moments():
    estimator = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0),
        batch_size=1,
    )
    estimator.observe_group(
        [
            TetherRegressionStats(
                weight=1,
                alpha_alpha=1e-300,
                rho_rho=1e-300,
                alpha_target=2e-300,
                rho_target=-1e-300,
                target_target=5e-300,
            )
        ]
    )

    assert estimator.coefficients == pytest.approx((2.0, -1.0))
    assert estimator.last_condition == pytest.approx(1.0)


def test_state_round_trip_preserves_pending_moments_and_diagnostics():
    config = AdaptiveTetherConfig(ridge=0.0, ema_decay=0.5)
    original = AdaptiveTetherCoefficients(config, batch_size=3)
    original.observe_group([_row(1.0, 0.0, 2.0), _row(0.0, 1.0, -1.0), _row(1.0, 1.0, 1.0)])
    original.observe_group([_row(2.0, 1.0, 3.0)])

    restored = AdaptiveTetherCoefficients(config, batch_size=3)
    restored.load_state_dict(original.state_dict())

    assert restored.state_dict() == original.state_dict()
    assert restored.metrics() == original.metrics()


def test_checkpoint_rejects_a_changed_regression_batch_size():
    original = AdaptiveTetherCoefficients(AdaptiveTetherConfig(), batch_size=2)
    restored = AdaptiveTetherCoefficients(AdaptiveTetherConfig(), batch_size=3)
    with pytest.raises(ValueError, match="batch_size"):
        restored.load_state_dict(original.state_dict())


def test_checkpoint_rejects_a_changed_reward_range():
    original = AdaptiveTetherCoefficients(AdaptiveTetherConfig(), batch_size=2)
    restored = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(),
        batch_size=2,
        reward_range=(0.0, 2.0),
    )
    with pytest.raises(ValueError, match="reward_range"):
        restored.load_state_dict(original.state_dict())
