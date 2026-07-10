from dataclasses import asdict

import pytest

from prime_rl.configs.algorithm import AdaptiveTetherConfig, TetherBaselineConfig
from prime_rl.orchestrator.algo.tether import (
    AdaptiveTetherCoefficients,
    TetherCoefficientTable,
    TetherRegressionStats,
    TetherRolloutStats,
    TetherRuntime,
    tether_branch_advantages,
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


def _rollout(*bins: TetherRegressionStats) -> TetherRolloutStats:
    return TetherRolloutStats(bins)


def _global(*rows: TetherRegressionStats) -> list[TetherRolloutStats]:
    return [_rollout(row) for row in rows]


def _repeat(row: TetherRegressionStats, count: int) -> TetherRegressionStats:
    total = TetherRegressionStats()
    for _ in range(count):
        total += row
    return total


def _branch_stats(
    *,
    reward: float,
    group_anchor: float,
    values: list[float],
    reward_range: tuple[float, float] = (0.0, 1.0),
) -> TetherRegressionStats:
    _, bins = tether_branch_advantages(
        reward=reward,
        group_anchor=group_anchor,
        values=values,
        train_mask=[True] * len(values),
        action_mask=[True] * len(values),
        coefficients=TetherCoefficientTable((0.0,), (0.0,)),
        reward_range=reward_range,
    )
    assert bins is not None
    return bins[0]


def test_joint_regression_recovers_two_factors_exactly():
    estimator = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0),
        batch_size=3,
    )
    estimator.observe_group(_global(_row(1.0, 0.0, 2.0), _row(0.0, 1.0, -1.0), _row(1.0, 1.0, 1.0)))

    assert estimator.coefficients == pytest.approx((2.0, -1.0))
    assert estimator.last_mse_fit == pytest.approx(0.0, abs=1e-12)
    assert estimator.updates == 1


def test_zero_initialized_ema_is_intentionally_not_bias_corrected():
    estimator = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.9),
        batch_size=3,
    )
    estimator.observe_group(_global(_row(1.0, 0.0, 2.0), _row(0.0, 1.0, -1.0), _row(1.0, 1.0, 1.0)))

    assert estimator.coefficients == pytest.approx((0.2, -0.1))


def test_rollout_moments_are_token_weighted_and_branch_local():
    first = _branch_stats(reward=1.0, group_anchor=0.5, values=[0.7, 1.0])
    second = _branch_stats(reward=1.0, group_anchor=0.5, values=[0.2, 0.4])
    stats = first + second

    # B_loo=.5. Branch features are (.2, [0,.3]) and (-.3, [0,.2]); y=.5.
    assert stats.weight == 4
    assert stats.alpha_alpha == pytest.approx(2 * 0.2**2 + 2 * 0.3**2)
    assert stats.alpha_rho == pytest.approx(0.2 * 0.3 - 0.3 * 0.2)
    assert stats.rho_rho == pytest.approx(0.3**2 + 0.2**2)
    assert stats.alpha_target == pytest.approx(2 * 0.2 * 0.5 - 2 * 0.3 * 0.5)
    assert stats.rho_target == pytest.approx((0.3 + 0.2) * 0.5)
    assert stats.target_target == pytest.approx(4 * 0.5**2)


def test_native_action_provenance_defines_start_value_and_position():
    advantages, bins = tether_branch_advantages(
        reward=1.0,
        group_anchor=0.5,
        values=[9.0, 0.7, 8.0, 0.9],
        train_mask=[False, False, False, True],
        action_mask=[False, True, False, True],
        coefficients=TetherCoefficientTable((0.0, 1.0), (0.0, 0.0), bin_size=1),
        reward_range=(0.0, 1.0),
    )

    # The shared first action is not trained in this branch, but still supplies
    # V0=.7 and advances the unique action to bin 1.
    assert advantages == pytest.approx([0.0, 0.0, 0.0, 0.3])
    assert bins[0].weight == 0
    assert bins[1].weight == 1
    assert bins[1].alpha_target == pytest.approx(0.1)


def test_context_tokens_do_not_advance_action_position():
    advantages, bins = tether_branch_advantages(
        reward=1.0,
        group_anchor=0.5,
        values=[float("nan"), 0.7, float("nan"), 1.0],
        train_mask=[False, True, False, True],
        action_mask=[False, True, False, True],
        coefficients=TetherCoefficientTable((0.0, 1.0), (0.0, 0.0), bin_size=1),
        reward_range=(0.0, 1.0),
    )

    assert advantages == pytest.approx([0.0, 0.5, 0.0, 0.3])
    assert [stats.weight for stats in bins] == [1, 1]


def test_position_bin_boundaries_and_overflow_are_fixed_ex_ante():
    table = TetherCoefficientTable((0.0, 0.5, 1.0), (0.0, 0.5, 1.0), bin_size=256)

    assert [table.bin_index(position) for position in (0, 255, 256, 511, 512, 9999)] == [0, 0, 1, 1, 2, 2]


def test_tether_clips_the_complete_baseline():
    above, above_stats = tether_branch_advantages(
        reward=1.0,
        group_anchor=0.5,
        values=[9.0, 0.7, 8.0, 1.7],
        train_mask=[False, True, False, True],
        action_mask=[False, True, False, True],
        coefficients=TetherCoefficientTable((0.5,), (0.8,)),
        reward_range=(0.0, 1.0),
    )
    below, below_stats = tether_branch_advantages(
        reward=1.0,
        group_anchor=0.5,
        values=[9.0, 0.7, 8.0, -1.3],
        train_mask=[False, True, False, True],
        action_mask=[False, True, False, True],
        coefficients=TetherCoefficientTable((0.5,), (0.8,)),
        reward_range=(0.0, 1.0),
    )

    assert above == pytest.approx([0.0, 0.4, 0.0, 0.0])
    assert below == pytest.approx([0.0, 0.4, 0.0, 1.0])
    assert above_stats[0].clipped == below_stats[0].clipped == 1


def test_exact_rollout_batches_retain_overflow():
    estimator = AdaptiveTetherCoefficients(AdaptiveTetherConfig(), batch_size=2)
    estimator.observe_group(_global(_row(1.0, 0.0, 1.0), _row(-1.0, 0.0, -1.0), _row(0.0, 1.0, 2.0)))

    assert estimator.updates == 1
    assert estimator.pending_rollouts == 1
    estimator.observe_group(_global(_row(0.0, -1.0, -2.0)))
    assert estimator.updates == 2
    assert estimator.pending_rollouts == 0


def test_normalized_ridge_is_invariant_to_duplicating_every_row():
    rows = _global(_row(1.0, 0.0, 2.0), _row(0.0, 1.0, -1.0), _row(1.0, 1.0, 1.0))
    once = AdaptiveTetherCoefficients(AdaptiveTetherConfig(ridge=0.2, ema_decay=0.0), batch_size=3)
    twice = AdaptiveTetherCoefficients(AdaptiveTetherConfig(ridge=0.2, ema_decay=0.0), batch_size=6)
    once.observe_group(rows)
    twice.observe_group(rows + rows)

    assert twice.coefficients == pytest.approx(once.coefficients)


def test_reward_range_normalization_is_scale_invariant():
    unit = _branch_stats(reward=1.0, group_anchor=0.4, values=[0.7, 0.9])
    scaled = _branch_stats(
        reward=10.0,
        group_anchor=4.0,
        values=[7.0, 9.0],
        reward_range=(0.0, 10.0),
    )

    assert asdict(scaled) == pytest.approx(asdict(unit))


def test_singular_unregularized_batch_is_skipped_but_ridge_is_safe():
    no_ridge = AdaptiveTetherCoefficients(AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0), batch_size=1)
    no_ridge.observe_group(_global(_row(0.0, 0.0, 1.0)))
    assert no_ridge.updates == 0
    assert no_ridge.skipped_updates == 1
    assert no_ridge.last_condition == [float("inf")]
    assert "tether/condition_number" in no_ridge.metric_keys()
    assert "tether/condition_number" not in no_ridge.metrics()

    ridge = AdaptiveTetherCoefficients(AdaptiveTetherConfig(ridge=1e-6, ema_decay=0.0), batch_size=1)
    ridge.observe_group(_global(_row(0.0, 0.0, 1.0)))
    assert ridge.updates == 1
    assert ridge.coefficients == (0.0, 0.0)


def test_indefinite_moments_are_rejected_and_replace_stale_diagnostics():
    estimator = AdaptiveTetherCoefficients(AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0), batch_size=1)
    estimator.observe_group(
        _global(
            TetherRegressionStats(
                weight=1,
                alpha_alpha=1.0,
                alpha_rho=2.0,
                rho_rho=1.0,
                alpha_target=1.0,
                rho_target=1.0,
                target_target=1.0,
            )
        )
    )

    assert estimator.updates == 0
    assert estimator.skipped_updates == 1
    assert estimator.last_condition == [float("inf")]


def test_equilibrated_solve_handles_tiny_well_conditioned_moments():
    estimator = AdaptiveTetherCoefficients(AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0), batch_size=1)
    estimator.observe_group(
        _global(
            TetherRegressionStats(
                weight=1,
                alpha_alpha=1e-300,
                rho_rho=1e-300,
                alpha_target=2e-300,
                rho_target=-1e-300,
                target_target=5e-300,
            )
        )
    )

    assert estimator.coefficients == pytest.approx((2.0, -1.0))
    assert estimator.last_condition == pytest.approx([1.0])


def test_positioned_regression_recovers_independent_bin_coefficients():
    estimator = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0, min_bin_rollouts=1),
        batch_size=3,
        num_bins=2,
        bin_size=1,
        position_horizon=2,
    )
    estimator.observe_group(
        [
            _rollout(_row(1.0, 0.0, 2.0), _row(1.0, 0.0, 3.0)),
            _rollout(_row(0.0, 1.0, -1.0), _row(0.0, 1.0, 4.0)),
            _rollout(_row(1.0, 1.0, 1.0), _row(1.0, 1.0, 7.0)),
        ]
    )

    assert estimator.coefficient_table.alpha == pytest.approx((2.0, 3.0))
    assert estimator.coefficient_table.rho == pytest.approx((-1.0, 4.0))
    assert estimator.last_fit_valid == [True, True]


def test_positioned_ridge_fit_is_invariant_to_unrelated_bin_token_count():
    config = AdaptiveTetherConfig(ridge=0.2, ema_decay=0.0, min_bin_rollouts=1)

    def fit(other_bin_repeats: int) -> TetherCoefficientTable:
        estimator = AdaptiveTetherCoefficients(
            config,
            batch_size=3,
            num_bins=2,
            bin_size=1,
            position_horizon=2,
        )
        estimator.observe_group(
            [
                _rollout(_row(1.0, 0.0, 2.0), _repeat(_row(1.0, 0.0, 4.0), other_bin_repeats)),
                _rollout(_row(0.0, 1.0, -1.0), _repeat(_row(0.0, 1.0, 3.0), other_bin_repeats)),
                _rollout(_row(1.0, 1.0, 1.0), _repeat(_row(1.0, 1.0, 7.0), other_bin_repeats)),
            ]
        )
        return estimator.coefficient_table

    sparse = fit(1)
    dense = fit(100)

    assert dense.alpha[0] == pytest.approx(sparse.alpha[0])
    assert dense.rho[0] == pytest.approx(sparse.rho[0])


def test_sparse_bins_hold_state_and_use_support_adjusted_ema():
    estimator = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.81, min_bin_rollouts=2),
        batch_size=4,
        num_bins=2,
        bin_size=1,
        position_horizon=2,
    )
    empty = TetherRegressionStats()
    estimator.observe_group(
        [
            _rollout(empty, _row(1.0, 0.0, 2.0)),
            _rollout(empty, _row(0.0, 1.0, 3.0)),
            _rollout(empty, empty),
            _rollout(empty, empty),
        ]
    )

    # Two of four rollouts contribute, so effective decay is .81 ** (2/4) = .9.
    assert estimator.coefficient_table.alpha == pytest.approx((0.0, 0.2))
    assert estimator.coefficient_table.rho == pytest.approx((0.0, 0.3))
    assert estimator.last_fit_valid == [False, True]
    assert estimator.bin_updates == [0, 1]


def test_positioned_defaults_require_one_eighth_batch_support():
    estimator = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(),
        batch_size=256,
        num_bins=2,
        bin_size=1024,
        position_horizon=2048,
    )

    assert estimator.min_bin_rollouts == 32


def test_static_position_schedule_ramps_configured_endpoints():
    runtime = TetherRuntime(
        TetherBaselineConfig(alpha=1.0, rho=0.6, position={"bin_size": 2, "max_action_tokens": 6}),
        value_seq_len=8,
        policy_seq_len=8,
        adaptive_batch_size=None,
    )

    assert runtime.coefficients.alpha == pytest.approx((0.0, 0.5, 1.0))
    assert runtime.coefficients.rho == pytest.approx((0.0, 0.3, 0.6))


def test_position_configuration_requires_multiple_bounded_bins():
    with pytest.raises(ValueError, match="at least two bins"):
        TetherRuntime(
            TetherBaselineConfig(position={"bin_size": 8, "max_action_tokens": 8}),
            value_seq_len=8,
            policy_seq_len=8,
            adaptive_batch_size=None,
        )

    with pytest.raises(ValueError, match="maximum is 128"):
        TetherRuntime(
            TetherBaselineConfig(position={"bin_size": 1, "max_action_tokens": 129}),
            value_seq_len=129,
            policy_seq_len=129,
            adaptive_batch_size=None,
        )


def test_state_round_trip_preserves_positioned_pending_moments_and_diagnostics():
    config = AdaptiveTetherConfig(ridge=0.0, ema_decay=0.5, min_bin_rollouts=1)
    original = AdaptiveTetherCoefficients(
        config,
        batch_size=3,
        num_bins=2,
        bin_size=1,
        position_horizon=2,
    )
    original.observe_group(
        [
            _rollout(_row(1.0, 0.0, 2.0), _row(1.0, 0.0, 3.0)),
            _rollout(_row(0.0, 1.0, -1.0), _row(0.0, 1.0, 4.0)),
            _rollout(_row(1.0, 1.0, 1.0), _row(1.0, 1.0, 7.0)),
        ]
    )
    original.observe_group([_rollout(_row(2.0, 1.0, 3.0), TetherRegressionStats())])

    restored = AdaptiveTetherCoefficients(
        config,
        batch_size=3,
        num_bins=2,
        bin_size=1,
        position_horizon=2,
    )
    restored.load_state_dict(original.state_dict())

    assert original.last_fit_token_fraction == 1.0
    assert restored.state_dict() == original.state_dict()
    assert restored.metrics() == original.metrics()


def test_scalar_checkpoint_migrates_only_into_global_mode():
    legacy = {
        "batch_size": 2,
        "reward_range": (0.0, 1.0),
        "alpha": 0.2,
        "rho": -0.4,
        "pending": [asdict(_row(1.0, 0.0, 1.0))],
    }
    global_estimator = AdaptiveTetherCoefficients(AdaptiveTetherConfig(), batch_size=2)
    global_estimator.load_state_dict(legacy)

    assert global_estimator.coefficients == pytest.approx((0.2, -0.4))
    assert global_estimator.pending_rollouts == 1

    positioned = AdaptiveTetherCoefficients(
        AdaptiveTetherConfig(),
        batch_size=2,
        num_bins=2,
        bin_size=1,
        position_horizon=2,
    )
    with pytest.raises(ValueError, match="cannot initialize position bins"):
        positioned.load_state_dict(legacy)


def test_checkpoint_rejects_changed_position_contract():
    config = AdaptiveTetherConfig(min_bin_rollouts=1)
    original = AdaptiveTetherCoefficients(
        config,
        batch_size=2,
        num_bins=2,
        bin_size=1,
        position_horizon=2,
    )
    restored = AdaptiveTetherCoefficients(
        config,
        batch_size=2,
        num_bins=2,
        bin_size=2,
        position_horizon=4,
    )

    with pytest.raises(ValueError, match="checkpoint contract"):
        restored.load_state_dict(original.state_dict())


def test_checkpoint_rejects_changed_global_controller_contract():
    original = AdaptiveTetherCoefficients(AdaptiveTetherConfig(), batch_size=2)
    state = original.state_dict()

    with pytest.raises(ValueError, match="checkpoint contract"):
        AdaptiveTetherCoefficients(AdaptiveTetherConfig(), batch_size=3).load_state_dict(state)
    with pytest.raises(ValueError, match="checkpoint contract"):
        AdaptiveTetherCoefficients(
            AdaptiveTetherConfig(),
            batch_size=2,
            reward_range=(0.0, 2.0),
        ).load_state_dict(state)


def test_checkpoint_rejects_unknown_schema_and_coerced_scalar_state():
    estimator = AdaptiveTetherCoefficients(AdaptiveTetherConfig(), batch_size=2)
    state = estimator.state_dict()

    with pytest.raises(ValueError, match="unsupported.*schema"):
        estimator.load_state_dict(state | {"schema_version": 3})
    with pytest.raises(ValueError, match="invalid updates"):
        estimator.load_state_dict(state | {"updates": 1.5})
    with pytest.raises(ValueError, match="invalid last fit valid"):
        estimator.load_state_dict(state | {"last_fit_valid": ["false"]})
