import math

import pytest
import torch

from prime_rl.configs.algorithm import AdaptiveTetherConfig, LinearMixBaselineConfig, TetherBaselineConfig
from prime_rl.configs.value import ClassificationValueLossConfig, MSEValueLossConfig
from prime_rl.value.math import (
    align_value_logits,
    compute_value_loss,
    predict_values,
)


def test_align_value_logits_resets_at_packed_sequence_boundaries():
    logits = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])

    aligned = align_value_logits(logits, [2, 3])

    assert aligned.squeeze(-1).tolist() == [[0.0, 1.0, 0.0, 3.0, 4.0]]


def test_align_value_logits_keeps_zero_gradient_path_for_padding_rank():
    logits = torch.tensor([[[1.0]]], requires_grad=True)

    aligned = align_value_logits(logits, [1])
    aligned.sum().backward()

    assert logits.grad is not None
    assert logits.grad.item() == 0.0


def test_classification_value_prediction_is_support_expectation():
    config = ClassificationValueLossConfig(reward_range=(-1.0, 1.0), num_bins=3)
    logits = torch.tensor([[[0.0, 20.0, 0.0]]])

    assert predict_values(logits, config).item() == pytest.approx(0.0, abs=1e-6)


def test_regression_value_prediction_is_unbounded_without_sigmoid():
    logits = torch.tensor([[[2.5]]])

    assert predict_values(logits, MSEValueLossConfig()).item() == pytest.approx(2.5)


def test_classification_uses_expectation_preserving_two_hot_targets():
    logits = torch.tensor([[[math.log(0.6), math.log(0.4)]]], requires_grad=True)
    loss, metrics = compute_value_loss(
        logits=logits,
        targets=torch.tensor([[0.4]]),
        mask=torch.tensor([[True]]),
        config=ClassificationValueLossConfig(reward_range=(0.0, 1.0), num_bins=2),
        scale=1,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-6)
    assert metrics["value/error"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["value/entropy"].item() == pytest.approx(-0.6 * math.log(0.6) - 0.4 * math.log(0.4))
    assert metrics["value/confidence"].item() == pytest.approx(0.6)


def test_classification_rejects_targets_outside_support():
    with pytest.raises(ValueError, match="outside reward_range"):
        compute_value_loss(
            logits=torch.zeros(1, 1, 2),
            targets=torch.tensor([[1.1]]),
            mask=torch.tensor([[True]]),
            config=ClassificationValueLossConfig(reward_range=(0.0, 1.0), num_bins=2),
            scale=1,
        )


def test_value_loss_masks_context_tokens():
    loss, metrics = compute_value_loss(
        logits=torch.tensor([[[1.0], [100.0]]]),
        targets=torch.tensor([[0.0, 0.0]]),
        mask=torch.tensor([[True, False]]),
        config=MSEValueLossConfig(),
        scale=1,
    )

    assert loss.item() == pytest.approx(1.0)
    assert metrics["value/loss"].tolist() == [1.0]
    assert metrics["value/error"].tolist() == [1.0]
    assert metrics["value/squared_error"].tolist() == [1.0]


def test_mixed_baselines_default_to_leave_one_out_and_allow_mean():
    assert LinearMixBaselineConfig().group == "leave_one_out"
    assert TetherBaselineConfig().group == "leave_one_out"
    assert LinearMixBaselineConfig(group="mean").group == "mean"
    assert TetherBaselineConfig(group="mean").group == "mean"


def test_mixed_baseline_coefficients_allow_any_finite_value():
    assert LinearMixBaselineConfig(rho=-0.25).rho == -0.25
    tether = TetherBaselineConfig(alpha=-1.5, rho=2.0)
    assert tether.alpha == -1.5
    assert tether.rho == 2.0


def test_adaptive_tether_defaults_to_zero_initialized_slow_ema():
    adaptive = AdaptiveTetherConfig()
    assert adaptive.initial_alpha == 0.0
    assert adaptive.initial_rho == 0.0
    assert adaptive.ema_decay == 0.9
    assert adaptive.ridge == 1e-6
    assert adaptive.batch_size is None
    assert adaptive.min_bin_rollouts is None

    position = TetherBaselineConfig(position={}).position
    assert position is not None
    assert position.bin_size == 1024
    assert position.max_action_tokens is None


def test_adaptive_tether_requires_leave_one_out():
    with pytest.raises(ValueError, match="leave_one_out"):
        TetherBaselineConfig(group="mean", adaptive={})


def test_adaptive_tether_resolved_config_round_trips_static_defaults():
    baseline = TetherBaselineConfig(alpha=0.2, rho=0.3, adaptive={})
    reloaded = TetherBaselineConfig.model_validate(baseline.model_dump())

    assert reloaded.adaptive is not None
    assert reloaded.adaptive.initial_alpha == 0.0
    assert reloaded.adaptive.initial_rho == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ema_decay": -0.1},
        {"ema_decay": 1.0},
        {"ridge": -1e-6},
        {"min_bin_rollouts": 0},
        {"initial_alpha": float("nan")},
        {"initial_rho": float("inf")},
    ],
)
def test_adaptive_tether_rejects_invalid_fit_parameters(kwargs):
    with pytest.raises(ValueError):
        AdaptiveTetherConfig(**kwargs)


@pytest.mark.parametrize(
    ("config_type", "kwargs"),
    [
        (LinearMixBaselineConfig, {"rho": float("inf")}),
        (TetherBaselineConfig, {"alpha": float("nan")}),
        (TetherBaselineConfig, {"rho": float("-inf")}),
    ],
)
def test_mixed_baseline_coefficients_reject_nonfinite_values(config_type, kwargs):
    with pytest.raises(ValueError):
        config_type(**kwargs)


def test_bounded_value_configs_reject_nonfinite_reward_ranges():
    with pytest.raises(ValueError, match="reward_range"):
        ClassificationValueLossConfig(reward_range=(float("nan"), 1.0))
    with pytest.raises(ValueError, match="reward_range"):
        TetherBaselineConfig(reward_range=(0.0, float("inf")))
    with pytest.raises(ValueError, match="reward_range"):
        TetherBaselineConfig(reward_range=(-1e308, 1e308))
