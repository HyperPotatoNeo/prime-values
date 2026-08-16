import asyncio
from unittest.mock import MagicMock

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import (
    AdaptiveTetherConfig,
    GRPOAlgoConfig,
    TetherBaselineConfig,
)
from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.orchestrator.algo.advantage import compute_gae
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.tether import (
    AdaptivePositionTetherCoefficients,
    AdaptiveTetherCoefficient,
    TetherCoefficientTable,
    TetherRegressionStats,
    TetherRolloutStats,
    TetherRuntime,
)
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout
from prime_rl.orchestrator.value_context import TokenPrefix


def _tether_row(feature: float, target: float) -> TetherRegressionStats:
    return TetherRegressionStats(
        weight=1,
        feature_feature=feature * feature,
        feature_target=feature * target,
        target_target=target * target,
    )


def _tether_rollout(*bins: TetherRegressionStats) -> TetherRolloutStats:
    return TetherRolloutStats(bins)


def _rollout(reward: float, *, values: list[float]) -> Rollout:
    nodes = [
        vf.MessageNode(
            message=vf.UserMessage(content="q"),
            token_ids=[0],
            mask=[False],
            logprobs=[0.0],
            sampled=False,
            parent=None,
        ),
        vf.MessageNode(
            message=vf.AssistantMessage(content="a"),
            token_ids=list(range(1, len(values))),
            mask=[True] * (len(values) - 1),
            logprobs=[-0.1] * (len(values) - 1),
            sampled=True,
            parent=0,
        ),
    ]
    rollout = Rollout(
        task=vf.Task(idx=0, prompt=None),
        nodes=nodes,
        rewards={"reward": reward},
        env_name="test",
    )
    rollout.samples = trace_to_samples(rollout, env_name="test")
    rollout.value_predictions = [values]
    return rollout


def _position_rollout(reward: float, *, values: list[float]) -> Rollout:
    if len(values) != 6:
        raise ValueError("position test rollout needs six token values")
    nodes = [
        vf.MessageNode(
            message=vf.UserMessage(content="q"),
            token_ids=[0],
            mask=[False],
            logprobs=[0.0],
            sampled=False,
            parent=None,
        ),
        vf.MessageNode(
            message=vf.AssistantMessage(content="a0"),
            token_ids=[1],
            mask=[True],
            logprobs=[-0.1],
            sampled=True,
            parent=0,
        ),
        vf.MessageNode(
            message=vf.UserMessage(content="observation"),
            token_ids=[2],
            mask=[False],
            logprobs=[0.0],
            sampled=False,
            parent=1,
        ),
        vf.MessageNode(
            message=vf.AssistantMessage(content="a1 a2 a3"),
            token_ids=[3, 4, 5],
            mask=[True, True, True],
            logprobs=[-0.1, -0.1, -0.1],
            sampled=True,
            parent=2,
        ),
    ]
    rollout = Rollout(
        task=vf.Task(idx=0, prompt=None),
        nodes=nodes,
        rewards={"reward": reward},
        env_name="test",
    )
    rollout.samples = trace_to_samples(rollout, env_name="test")
    rollout.value_predictions = [values]
    return rollout


def _attach_policy_gae(group: list[Rollout], *, gamma: float, gae_lambda: float) -> None:
    for rollout in group:
        assert rollout.value_predictions is not None
        rollout.value_advantages = []
        for sample, values in zip(rollout.samples, rollout.value_predictions, strict=True):
            advantages, _ = compute_gae(
                reward=float(rollout.reward),
                values=values,
                mask=sample.mask,
                gamma=gamma,
                gae_lambda=gae_lambda,
                value_target_lambda=gae_lambda,
            )
            rollout.value_advantages.append(advantages)


def test_tether_fit_recovers_unbounded_coefficient_and_default_ema_is_lagged():
    exact = AdaptiveTetherCoefficient(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0),
        batch_size=3,
    )
    rows = [_tether_row(1.0, -2.0), _tether_row(-2.0, 4.0), _tether_row(0.5, -1.0)]
    exact.observe_group(rows)
    assert exact.rho == pytest.approx(-2.0)

    default = AdaptiveTetherCoefficient(AdaptiveTetherConfig(ridge=0.0), batch_size=3)
    positive_rows = [_tether_row(1.0, 2.0), _tether_row(-2.0, -4.0), _tether_row(0.5, 1.0)]
    default.observe_group(positive_rows)
    assert default.rho == pytest.approx(0.1)
    assert default.last_fit_rho == pytest.approx(2.0)


def test_tether_relative_ridge_is_scale_and_duplicate_invariant():
    config = AdaptiveTetherConfig(ridge=0.2, ema_decay=0.0)
    rows = [_tether_row(1.0, 2.0), _tether_row(-2.0, -3.0)]
    scaled = [_tether_row(10.0, 20.0), _tether_row(-20.0, -30.0)]
    once = AdaptiveTetherCoefficient(config, batch_size=2)
    twice = AdaptiveTetherCoefficient(config, batch_size=4)
    rescaled = AdaptiveTetherCoefficient(config, batch_size=2)
    once.observe_group(rows)
    twice.observe_group(rows + rows)
    rescaled.observe_group(scaled)
    assert twice.rho == pytest.approx(once.rho)
    assert rescaled.rho == pytest.approx(once.rho)


def test_tether_exact_rollout_windows_retain_overflow_and_skip_zero_features():
    estimator = AdaptiveTetherCoefficient(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0),
        batch_size=2,
    )
    estimator.observe_group([_tether_row(1.0, 2.0), _tether_row(-1.0, -2.0), _tether_row(2.0, 4.0)])
    assert estimator.updates == 1
    assert estimator.pending_rollouts == 1
    estimator.observe_group([_tether_row(-2.0, -4.0)])
    assert estimator.updates == 2
    assert estimator.pending_rollouts == 0

    degenerate = AdaptiveTetherCoefficient(AdaptiveTetherConfig(), batch_size=1)
    degenerate.observe_group([_tether_row(0.0, 1.0)])
    assert degenerate.updates == 0
    assert degenerate.skipped_updates == 1
    assert not degenerate.last_fit_valid
    assert degenerate.last_mse_fit == pytest.approx(1.0)
    assert degenerate.last_mse_ema == pytest.approx(1.0)


def test_tether_position_table_boundaries_and_overflow_are_fixed_ex_ante():
    table = TetherCoefficientTable((0.0, 0.5, 1.0), bin_size=2)

    assert [table.bin_index(position) for position in (0, 1, 2, 3, 4, 999)] == [0, 0, 1, 1, 2, 2]


def test_positioned_tether_fit_recovers_independent_bin_coefficients():
    estimator = AdaptivePositionTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0),
        batch_size=2,
        gamma=1.0,
        gae_lambda=0.5,
        num_bins=2,
        bin_size=2,
        min_bin_rollouts=1,
    )
    estimator.observe_group(
        [
            _tether_rollout(_tether_row(1.0, 2.0), _tether_row(2.0, -1.0)),
            _tether_rollout(_tether_row(-1.0, -2.0), _tether_row(-2.0, 1.0)),
        ]
    )

    assert estimator.coefficient_table.rho == pytest.approx((2.0, -0.5))
    assert estimator.last_fit_valid == [True, True]
    assert estimator.last_fit_token_fraction == 1.0


def test_positioned_tether_sparse_bins_hold_state_and_use_ordinary_ema():
    estimator = AdaptivePositionTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.5),
        batch_size=1,
        gamma=1.0,
        gae_lambda=0.5,
        num_bins=2,
        bin_size=1,
        min_bin_rollouts=1,
    )
    empty = TetherRegressionStats()

    estimator.observe_group([_tether_rollout(_tether_row(1.0, 2.0), empty)])
    assert estimator.coefficient_table.rho == pytest.approx((1.0, 0.0))
    estimator.observe_group([_tether_rollout(empty, _tether_row(2.0, -2.0))])
    assert estimator.coefficient_table.rho == pytest.approx((1.0, -0.5))
    estimator.observe_group([_tether_rollout(_tether_row(1.0, 4.0), empty)])
    assert estimator.coefficient_table.rho == pytest.approx((2.5, -0.5))
    assert estimator.bin_updates == [2, 1]


def test_positioned_tether_support_counts_distinct_rollouts_not_tokens():
    estimator = AdaptivePositionTetherCoefficients(
        AdaptiveTetherConfig(ridge=0.0, ema_decay=0.0),
        batch_size=2,
        gamma=1.0,
        gae_lambda=0.5,
        num_bins=2,
        bin_size=1,
        min_bin_rollouts=2,
    )
    many_rows_one_rollout = TetherRegressionStats(
        weight=10,
        feature_feature=10.0,
        feature_target=20.0,
        target_target=40.0,
    )
    estimator.observe_group(
        [
            _tether_rollout(many_rows_one_rollout, _tether_row(1.0, 3.0)),
            _tether_rollout(TetherRegressionStats(), _tether_row(-1.0, -3.0)),
        ]
    )

    assert estimator.coefficient_table.rho == pytest.approx((0.0, 3.0))
    assert estimator.last_fit_valid == [False, True]
    assert estimator.last_contributors == [1, 2]


def test_positioned_tether_state_round_trip_preserves_partial_window_and_contract():
    config = AdaptiveTetherConfig(ridge=0.0, ema_decay=0.5)
    original = AdaptivePositionTetherCoefficients(
        config,
        batch_size=2,
        gamma=0.9,
        gae_lambda=0.8,
        num_bins=2,
        bin_size=2,
        min_bin_rollouts=1,
    )
    original.observe_group(
        [
            _tether_rollout(_tether_row(1.0, 2.0), _tether_row(2.0, -1.0)),
            _tether_rollout(_tether_row(-1.0, -2.0), _tether_row(-2.0, 1.0)),
        ]
    )
    original.observe_group([_tether_rollout(_tether_row(3.0, 4.0), TetherRegressionStats())])

    restored = AdaptivePositionTetherCoefficients(
        config,
        batch_size=2,
        gamma=0.9,
        gae_lambda=0.8,
        num_bins=2,
        bin_size=2,
        min_bin_rollouts=1,
    )
    keys_before = restored.metric_keys()
    restored.load_state_dict(original.state_dict())

    assert restored.state_dict() == original.state_dict()
    assert restored.metrics() == original.metrics()
    assert restored.metric_keys() == keys_before
    assert len(restored.metrics()) == 15 + 3 * restored.num_bins
    assert not any("/batch_fit_rho" in key or "/tokens" in key for key in restored.metrics())

    changed_bins = AdaptivePositionTetherCoefficients(
        config,
        batch_size=2,
        gamma=0.9,
        gae_lambda=0.8,
        num_bins=3,
        bin_size=2,
        min_bin_rollouts=1,
    )
    with pytest.raises(ValueError, match="checkpoint contract"):
        changed_bins.load_state_dict(original.state_dict())
    with pytest.raises(ValueError, match="checkpoint"):
        restored.load_state_dict(
            AdaptiveTetherCoefficient(config, batch_size=2, gamma=0.9, gae_lambda=0.8).state_dict()
        )
    corrupt_support = original.state_dict()
    corrupt_support["pending_contributors"][0] = 0
    with pytest.raises(ValueError, match="inconsistent pending bin state"):
        restored.load_state_dict(corrupt_support)
    corrupt_empty = original.state_dict()
    corrupt_empty["pending_bins"][1]["feature_target"] = 1.0
    with pytest.raises(ValueError, match="inconsistent pending bin state"):
        restored.load_state_dict(corrupt_empty)


def test_tether_state_round_trip_preserves_partial_window_and_rejects_temporal_change():
    config = AdaptiveTetherConfig(ridge=0.0, ema_decay=0.5)
    original = AdaptiveTetherCoefficient(config, batch_size=2, gamma=0.9, gae_lambda=0.8)
    original.observe_group([_tether_row(1.0, 2.0)])
    state = original.state_dict()

    restored = AdaptiveTetherCoefficient(config, batch_size=2, gamma=0.9, gae_lambda=0.8)
    restored.load_state_dict(state)
    restored.observe_group([_tether_row(-1.0, -2.0)])
    assert restored.rho == pytest.approx(1.0)
    assert restored.updates == 1

    with pytest.raises(ValueError, match="contract"):
        AdaptiveTetherCoefficient(config, batch_size=2, gamma=0.8, gae_lambda=0.8).load_state_dict(state)
    with pytest.raises(ValueError, match="contract"):
        AdaptiveTetherCoefficient(config, batch_size=2, gamma=0.9, gae_lambda=0.7).load_state_dict(state)


def test_tether_rejects_incompatible_checkpoint_schema():
    config = AdaptiveTetherConfig(ridge=0.0, ema_decay=0.5)
    original = AdaptiveTetherCoefficient(config, batch_size=2, gamma=0.9, gae_lambda=1.0)
    state = original.state_dict()
    state["schema_version"] = 1
    with pytest.raises(ValueError, match="unsupported"):
        AdaptiveTetherCoefficient(config, batch_size=2, gamma=0.9, gae_lambda=1.0).load_state_dict(state)


@pytest.mark.parametrize(
    ("group_anchor", "success", "failure"),
    [
        ("leave_one_out", [0.0, 0.95, 0.9], [0.0, -0.95, -0.9]),
        ("mean", [0.0, 0.575, 0.525], [0.0, -0.575, -0.525]),
    ],
)
def test_tether_static_matches_mixture_formula_at_unit_gamma_lambda(group_anchor, success, failure):
    group = [
        _rollout(1.0, values=[9.0, 0.2, 0.4]),
        _rollout(0.0, values=[9.0, 0.8, 0.6]),
    ]
    _attach_policy_gae(group, gamma=1.0, gae_lambda=1.0)
    runtime = TetherRuntime(
        TetherBaselineConfig(adaptive=None, rho=0.25, group=group_anchor),
        gamma=1.0,
        gae_lambda=1.0,
        value_seq_len=3,
        policy_seq_len=3,
        adaptive_batch_size=None,
    )
    runtime.score_group(group)
    assert group[0].advantages == pytest.approx(success)
    assert group[1].advantages == pytest.approx(failure)


@pytest.mark.parametrize(
    ("gae_lambda", "success", "failure"),
    [
        (0.5, [0.0, 0.525, 0.0, 0.65, 0.45, 0.4], [0.0, -0.525, 0.0, -0.65, -0.45, -0.4]),
        (1.0, [0.0, 0.95, 0.0, 0.9, 0.55, 0.4], [0.0, -0.95, 0.0, -0.9, -0.55, -0.4]),
    ],
)
def test_positioned_tether_uses_native_action_bins_without_changing_q_lambda(
    gae_lambda,
    success,
    failure,
):
    group = [
        _position_rollout(1.0, values=[9.0, 0.2, 8.0, 0.4, 0.6, 0.8]),
        _position_rollout(0.0, values=[9.0, 0.8, 8.0, 0.6, 0.4, 0.2]),
    ]
    _attach_policy_gae(group, gamma=1.0, gae_lambda=gae_lambda)
    runtime = TetherRuntime(
        TetherBaselineConfig(
            adaptive=AdaptiveTetherConfig(
                batch_size=2,
                position={"bin_size": 2, "max_action_tokens": 4, "min_bin_rollouts": 1},
            )
        ),
        gamma=1.0,
        gae_lambda=gae_lambda,
        value_seq_len=6,
        policy_seq_len=6,
        adaptive_batch_size=2,
    )
    assert runtime.positioned_adaptive is not None
    runtime.positioned_adaptive._rho = [0.25, 0.75]

    runtime.score_group(group)

    assert group[0].advantages == pytest.approx(success)
    assert group[1].advantages == pytest.approx(failure)


def test_positioned_tether_runtime_fits_distinct_rhos_from_fixed_policy_lambda_return():
    group = [
        _position_rollout(1.0, values=[9.0, 0.2, 8.0, 0.4, 0.6, 0.8]),
        _position_rollout(0.0, values=[9.0, 0.8, 8.0, 0.6, 0.4, 0.2]),
    ]
    _attach_policy_gae(group, gamma=1.0, gae_lambda=0.5)
    runtime = TetherRuntime(
        TetherBaselineConfig(
            adaptive=AdaptiveTetherConfig(
                batch_size=2,
                ridge=0.0,
                ema_decay=0.0,
                position={"bin_size": 2, "max_action_tokens": 4, "min_bin_rollouts": 1},
            )
        ),
        gamma=1.0,
        gae_lambda=0.5,
        value_seq_len=6,
        policy_seq_len=6,
        adaptive_batch_size=2,
    )

    runtime.score_group(group)

    assert runtime.positioned_adaptive is not None
    assert runtime.positioned_adaptive.coefficient_table.rho == pytest.approx((2.075, 1.34))
    assert group[0].advantages == pytest.approx([0.0, 0.575, 0.0, 0.75, 0.9, 1.0])
    assert group[1].advantages == pytest.approx([0.0, -0.575, 0.0, -0.75, -0.9, -1.0])


def test_positioned_tether_rho_one_preserves_raw_gae_without_value_reconstruction():
    group = [
        _position_rollout(1.0, values=[9.0, 1e16, 8.0, -1e16, 1e16, -1e16]),
        _position_rollout(0.0, values=[9.0, -1e16, 8.0, 1e16, -1e16, 1e16]),
    ]
    raw_advantages = [
        [0.0, 1.0, 0.0, -1.0, 2.0, -2.0],
        [0.0, -1.0, 0.0, 1.0, -2.0, 2.0],
    ]
    for rollout, advantages in zip(group, raw_advantages, strict=True):
        rollout.value_advantages = [advantages]
    runtime = TetherRuntime(
        TetherBaselineConfig(
            adaptive=AdaptiveTetherConfig(
                batch_size=2,
                position={"bin_size": 2, "max_action_tokens": 4, "min_bin_rollouts": 1},
            )
        ),
        gamma=1.0,
        gae_lambda=0.5,
        value_seq_len=6,
        policy_seq_len=6,
        adaptive_batch_size=2,
    )
    assert runtime.positioned_adaptive is not None
    runtime.positioned_adaptive._rho = [1.0, 1.0]

    runtime.score_group(group)

    assert group[0].advantages == raw_advantages[0]
    assert group[1].advantages == raw_advantages[1]


def test_tether_rho_one_preserves_raw_gae_without_value_reconstruction():
    group = [
        _rollout(1.0, values=[9.0, 1e16, -1e16]),
        _rollout(0.0, values=[9.0, -1e16, 1e16]),
    ]
    raw_advantages = [[0.0, 1.0, -1.0], [0.0, -1.0, 1.0]]
    for rollout, advantages in zip(group, raw_advantages, strict=True):
        rollout.value_advantages = [advantages]
    runtime = TetherRuntime(
        TetherBaselineConfig(adaptive=None, rho=1.0),
        gamma=1.0,
        gae_lambda=0.5,
        value_seq_len=3,
        policy_seq_len=3,
        adaptive_batch_size=None,
    )

    runtime.score_group(group)

    assert group[0].advantages == raw_advantages[0]
    assert group[1].advantages == raw_advantages[1]


@pytest.mark.parametrize(
    ("rho", "success", "failure"),
    [
        (0.0, [0.0, 0.7, 1.0], [0.0, -0.7, -1.0]),
        (0.25, [0.0, 0.65, 0.9], [0.0, -0.65, -0.9]),
        (1.0, [0.0, 0.5, 0.6], [0.0, -0.5, -0.6]),
    ],
)
def test_tether_subtracts_mixed_baseline_from_fixed_value_lambda_return(rho, success, failure):
    group = [
        _rollout(1.0, values=[9.0, 0.2, 0.4]),
        _rollout(0.0, values=[9.0, 0.8, 0.6]),
    ]
    _attach_policy_gae(group, gamma=1.0, gae_lambda=0.5)
    runtime = TetherRuntime(
        TetherBaselineConfig(adaptive=None, rho=rho),
        gamma=1.0,
        gae_lambda=0.5,
        value_seq_len=3,
        policy_seq_len=3,
        adaptive_batch_size=None,
    )
    runtime.score_group(group)
    assert group[0].advantages == pytest.approx(success)
    assert group[1].advantages == pytest.approx(failure)


@pytest.mark.parametrize(
    ("gamma", "gae_lambda", "expected_advantage", "expected_rho"),
    [
        (0.5, 1.0, 0.5, 2.0),
        (1.0, 0.5, 0.75, 1.8),
    ],
)
@pytest.mark.parametrize("positioned", [False, True])
def test_tether_actor_cutoff_does_not_create_a_false_temporal_terminal(
    gamma,
    gae_lambda,
    expected_advantage,
    expected_rho,
    positioned,
):
    group = [
        _rollout(1.0, values=[9.0, 0.25, 0.5]),
        _rollout(0.0, values=[9.0, 0.5, 0.5]),
    ]
    _attach_policy_gae(group, gamma=gamma, gae_lambda=gae_lambda)
    adaptive = AdaptiveTetherConfig(
        batch_size=2,
        ridge=0.0,
        ema_decay=0.0,
        position={"bin_size": 1, "max_action_tokens": 2, "min_bin_rollouts": 1} if positioned else None,
    )
    runtime = TetherRuntime(
        TetherBaselineConfig(adaptive=adaptive),
        gamma=gamma,
        gae_lambda=gae_lambda,
        value_seq_len=3,
        policy_seq_len=2,
        adaptive_batch_size=2,
    )
    runtime.score_group(group)
    assert group[0].advantages == pytest.approx([0.0, expected_advantage, 0.0])
    if positioned:
        assert runtime.positioned_adaptive is not None
        assert runtime.positioned_adaptive.coefficient_table.rho[0] == pytest.approx(expected_rho)
    else:
        assert runtime.rho == pytest.approx(expected_rho)


def test_tether_adaptive_group_is_scored_before_its_fit_is_applied():
    group = [
        _rollout(1.0, values=[9.0, 0.5, 0.5]),
        _rollout(0.0, values=[9.0, 0.5, 0.5]),
    ]
    _attach_policy_gae(group, gamma=1.0, gae_lambda=1.0)
    runtime = TetherRuntime(
        TetherBaselineConfig(adaptive=AdaptiveTetherConfig(batch_size=2, ridge=0.0, ema_decay=0.0)),
        gamma=1.0,
        gae_lambda=1.0,
        value_seq_len=3,
        policy_seq_len=3,
        adaptive_batch_size=2,
    )
    runtime.score_group(group)
    assert group[0].advantages == pytest.approx([0.0, 1.0, 1.0])
    assert group[1].advantages == pytest.approx([0.0, -1.0, -1.0])
    assert runtime.rho == pytest.approx(2.0)


def test_tether_warmup_scores_rejected_groups_without_fitting_them():
    baseline = TetherBaselineConfig(adaptive=AdaptiveTetherConfig(batch_size=2, ridge=0.0, ema_decay=0.0))
    algorithm = GRPOAlgorithm(
        GRPOAlgoConfig(baseline=baseline),
        MagicMock(),
        value_evaluator=MagicMock(),
        value_config=ValueFunctionConfig(
            model={"seq_len": 3, "attn": "sdpa"},
            batch_size=2,
            warmup_updates=1,
        ),
        policy_seq_len=3,
    )

    rejected = [
        _rollout(1.0, values=[9.0, 0.5, 0.5]),
        _rollout(0.0, values=[9.0, 0.5, 0.5]),
    ]
    _attach_policy_gae(rejected, gamma=1.0, gae_lambda=1.0)
    for rollout in rejected:
        rollout.value_version = 0
    asyncio.run(algorithm.score_group(rejected))

    assert isinstance(algorithm.baseline_runtime, TetherRuntime)
    assert rejected[0].advantages == pytest.approx([0.0, 1.0, 1.0])
    assert rejected[1].advantages == pytest.approx([0.0, -1.0, -1.0])
    assert algorithm.baseline_runtime.rho == 0.0
    assert algorithm.baseline_runtime.adaptive is not None
    assert algorithm.baseline_runtime.adaptive.pending_rollouts == 0

    accepted = [
        _rollout(1.0, values=[9.0, 0.5, 0.5]),
        _rollout(0.0, values=[9.0, 0.5, 0.5]),
    ]
    _attach_policy_gae(accepted, gamma=1.0, gae_lambda=1.0)
    for rollout in accepted:
        rollout.value_version = 1
    asyncio.run(algorithm.score_group(accepted))

    assert algorithm.baseline_runtime.rho == pytest.approx(2.0)


def test_tether_fit_uses_policy_lambda_return_not_critic_targets():
    group = [
        _rollout(1.0, values=[9.0, 0.25, 0.5]),
        _rollout(0.0, values=[9.0, 0.5, 0.5]),
    ]
    baseline = TetherBaselineConfig(adaptive=AdaptiveTetherConfig(batch_size=2, ridge=0.0, ema_decay=0.0))
    value_config = ValueFunctionConfig(
        model={"seq_len": 3, "attn": "sdpa"},
        batch_size=2,
        gamma=0.5,
        gae_lambda=0.3,
        value_target_lambda=0.7,
    )
    algorithm = GRPOAlgorithm(
        GRPOAlgoConfig(baseline=baseline),
        MagicMock(),
        value_evaluator=MagicMock(),
        value_config=value_config,
        policy_seq_len=3,
    )
    for rollout, values in zip(group, ([9.0, 0.25, 0.5], [9.0, 0.5, 0.5]), strict=True):
        algorithm._assign_value_result(rollout, [values], version=1)
    critic_returns = [[branch[:] for branch in rollout.value_returns or []] for rollout in group]

    asyncio.run(algorithm.score_group(group))

    assert isinstance(algorithm.baseline_runtime, TetherRuntime)
    assert algorithm.baseline_runtime.rho == pytest.approx(239 / 130)
    assert [rollout.value_returns for rollout in group] == critic_returns


def test_grpo_wraps_adaptive_state_and_rejects_static_restore():
    adaptive_baseline = TetherBaselineConfig()
    static_baseline = TetherBaselineConfig(adaptive=None)
    value_config = ValueFunctionConfig(model={"seq_len": 3, "attn": "sdpa"}, batch_size=2)

    def algorithm(baseline):
        return GRPOAlgorithm(
            GRPOAlgoConfig(baseline=baseline),
            MagicMock(),
            value_evaluator=MagicMock(),
            value_config=value_config,
            policy_seq_len=3,
        )

    state = algorithm(adaptive_baseline).state_dict()
    assert set(state) == {adaptive_baseline.type}
    algorithm(adaptive_baseline).load_state_dict(state)
    with pytest.raises(ValueError, match="adaptive mode is disabled"):
        algorithm(static_baseline).load_state_dict(state)


def test_positioned_tether_visibility_honors_critic_prefix_and_actor_limits():
    config = TetherBaselineConfig(
        adaptive=AdaptiveTetherConfig(
            batch_size=2,
            position={"bin_size": 1, "max_action_tokens": 2, "min_bin_rollouts": 1},
        )
    )

    def score(*, value_seq_len: int, policy_seq_len: int, conditioned: bool) -> list[float]:
        group = [
            _rollout(1.0, values=[9.0, 0.2, 0.6]),
            _rollout(0.0, values=[9.0, 0.8, 0.4]),
        ]
        _attach_policy_gae(group, gamma=1.0, gae_lambda=1.0)
        if conditioned:
            for rollout in group:
                rollout.value_prefix = TokenPrefix(token_ids=(99,), insert_at=0)
        runtime = TetherRuntime(
            config,
            gamma=1.0,
            gae_lambda=1.0,
            value_seq_len=value_seq_len,
            policy_seq_len=policy_seq_len,
            adaptive_batch_size=2,
        )
        assert runtime.positioned_adaptive is not None
        runtime.positioned_adaptive._rho = [0.0, 1.0]
        runtime.score_group(group)
        return group[0].advantages or []

    assert score(value_seq_len=2, policy_seq_len=3, conditioned=False) == pytest.approx([0.0, 1.0, 0.0])
    assert score(value_seq_len=2, policy_seq_len=3, conditioned=True) == pytest.approx([0.0, 1.0, 0.4])
    assert score(value_seq_len=3, policy_seq_len=2, conditioned=True) == pytest.approx([0.0, 1.0, 0.0])


@pytest.mark.parametrize(
    "runtime",
    [
        (
            TetherRuntime(
                TetherBaselineConfig(adaptive=None),
                gamma=1.0,
                gae_lambda=1.0,
                value_seq_len=3,
                policy_seq_len=3,
                adaptive_batch_size=None,
            )
        ),
        (
            TetherRuntime(
                TetherBaselineConfig(
                    adaptive=AdaptiveTetherConfig(
                        batch_size=2,
                        position={"bin_size": 1, "max_action_tokens": 2},
                    )
                ),
                gamma=1.0,
                gae_lambda=1.0,
                value_seq_len=3,
                policy_seq_len=3,
                adaptive_batch_size=2,
            )
        ),
    ],
)
def test_value_backed_runtimes_reject_short_padded_prediction_streams(runtime):
    group = [
        _rollout(1.0, values=[9.0, 0.2, 0.4]),
        _rollout(0.0, values=[9.0, 0.8, 0.6]),
    ]
    _attach_policy_gae(group, gamma=1.0, gae_lambda=1.0)
    group[0].value_predictions = [[9.0, 0.2]]
    with pytest.raises(ValueError, match="padded sample stream"):
        runtime.score_group(group)
