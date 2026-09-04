import math

import pytest
import torch

from prime_rl.configs.value import ClassificationValueLossConfig, MSEValueLossConfig
from prime_rl.value.math import (
    align_value_logits,
    compute_value_loss,
    predict_values,
)


@pytest.mark.parametrize("lengths", [[1], [1, 1, 1], [2, 3], [1, 4, 1, 17], [512] * 64])
@pytest.mark.parametrize("output_size", [1, 2, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_align_value_logits_resets_at_packed_sequence_boundaries(lengths, output_size, dtype):
    generator = torch.Generator().manual_seed(31)
    logits = torch.randn(1, sum(lengths), output_size, generator=generator, dtype=dtype, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    flat = reference_logits.reshape(-1, output_size)
    expected = flat * 0.0
    offset = 0
    for length in lengths:
        if length > 1:
            expected[offset + 1 : offset + length] = flat[offset : offset + length - 1]
        offset += length
    expected = expected.reshape_as(logits)

    aligned = align_value_logits(logits, lengths)
    torch.testing.assert_close(aligned, expected, rtol=0, atol=0)
    weights = torch.randn(logits.shape, generator=generator, dtype=dtype)
    (aligned * weights).sum().backward(retain_graph=True)
    (expected * weights).sum().backward(retain_graph=True)
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=0, atol=0)
    logits.grad = reference_logits.grad = None
    targets = torch.rand(logits.shape[:-1], generator=generator)
    mask = torch.ones_like(targets, dtype=torch.bool)
    offset = 0
    for length in lengths:
        mask[0, offset] = False
        offset += length
    config = MSEValueLossConfig() if output_size == 1 else ClassificationValueLossConfig(num_bins=output_size)
    loss, _ = compute_value_loss(aligned, targets, mask, config, int(mask.sum()))
    reference_loss, _ = compute_value_loss(expected, targets, mask, config, int(mask.sum()))
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=0, atol=0)


@pytest.mark.parametrize("lengths", [[1], [1, 1, 1]])
def test_align_value_logits_keeps_zero_gradient_path_for_padding_rank(lengths):
    logits = torch.ones(1, sum(lengths), 2, requires_grad=True)

    aligned = align_value_logits(logits, lengths)
    aligned.sum().backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_align_value_logits_preserves_nonfinite_boundary_values():
    logits = torch.tensor([[[float("nan")], [float("inf")], [-float("inf")]]])
    assert torch.isnan(align_value_logits(logits, [1, 1, 1])).all()


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


def test_bounded_value_configs_reject_nonfinite_reward_ranges():
    with pytest.raises(ValueError, match="reward_range"):
        ClassificationValueLossConfig(reward_range=(float("nan"), 1.0))
