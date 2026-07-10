import math

import pytest
import torch

from prime_rl.configs.algorithm import LinearMixBaselineConfig, TetherBaselineConfig
from prime_rl.configs.value import ClassificationValueLossConfig, MSEValueLossConfig
from prime_rl.value.math import (
    align_value_logits,
    compute_gae,
    compute_value_loss,
    group_advantages,
    linear_mix_advantages,
    predict_values,
    tether_advantages,
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


def test_gae_monte_carlo_terminal_reward_ignores_context_tokens():
    advantages, returns = compute_gae(
        reward=1.0,
        values=[9.0, 0.2, 8.0, 0.6],
        mask=[False, True, False, True],
        gamma=1.0,
        gae_lambda=1.0,
        value_target_lambda=1.0,
    )

    assert advantages == pytest.approx([0.0, 0.8, 0.0, 0.4])
    assert returns == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_gae_nontrivial_gamma_and_lambda():
    advantages, returns = compute_gae(
        reward=1.0,
        values=[9.0, 0.2, 8.0, 0.4],
        mask=[False, True, False, True],
        gamma=0.9,
        gae_lambda=0.5,
        value_target_lambda=0.5,
    )

    assert advantages == pytest.approx([0.0, 0.43, 0.0, 0.6])
    assert returns == pytest.approx([0.0, 0.63, 0.0, 1.0])


def test_policy_gae_and_value_target_use_independent_lambdas():
    advantages, returns = compute_gae(
        reward=1.0,
        values=[9.0, 0.2, 8.0, 0.4],
        mask=[False, True, False, True],
        gamma=1.0,
        gae_lambda=0.0,
        value_target_lambda=1.0,
    )

    assert advantages == pytest.approx([0.0, 0.2, 0.0, 0.6])
    assert returns == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_leave_one_out_group_advantage_excludes_own_reward():
    assert group_advantages([0.0, 1.0, 1.0], "leave_one_out") == pytest.approx([-1.0, 0.5, 0.5])


def test_leave_one_out_requires_siblings():
    with pytest.raises(ValueError, match="group_size"):
        group_advantages([1.0], "leave_one_out")


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


def test_linear_mix_uses_static_unbounded_coefficient_on_actions():
    output = linear_mix_advantages(
        group_advantage=1.0,
        value_advantages=[0.0, 0.0, 0.0, -1.0],
        mask=[False, True, False, True],
        config=LinearMixBaselineConfig(rho=2.0),
    )

    assert output == pytest.approx([0.0, -1.0, 0.0, -3.0])


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


def test_tether_clips_the_full_baseline_above_reward_range():
    output = tether_advantages(
        reward=1.0,
        group_advantage=0.5,
        values=[9.0, 0.7, 8.0, 1.7],
        mask=[False, True, False, True],
        config=TetherBaselineConfig(alpha=0.5, rho=0.8),
    )

    # B=0.5; first baseline=.6. The complete final expression is
    # .5 + .5(.7-.5) + .8(1.7-.7) = 1.4, then the full baseline clips to 1.
    assert output == pytest.approx([0.0, 0.4, 0.0, 0.0])


def test_tether_clips_the_full_baseline_below_reward_range():
    output = tether_advantages(
        reward=1.0,
        group_advantage=0.5,
        values=[9.0, 0.7, 8.0, -1.3],
        mask=[False, True, False, True],
        config=TetherBaselineConfig(alpha=0.5, rho=0.8),
    )

    # .5 + .5(.7-.5) + .8(-1.3-.7) = -1.0, then the full baseline clips to 0.
    assert output == pytest.approx([0.0, 0.4, 0.0, 1.0])
