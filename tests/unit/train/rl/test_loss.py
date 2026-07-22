import pytest
import torch

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, SPMALossConfig
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, compute_entropy, compute_loss, setup_rl_loss_fn
from prime_rl.trainer.rl.token_export import _compute_export_tensors

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    ref_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        advantages,
        loss_mask=loss_mask,
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )
    assert loss.shape == ()


def test_gspo_loss():
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    ref_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        advantages,
        loss_mask=loss_mask,
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )
    assert loss.shape == ()


def test_spma_loss_applies_bounded_linear_weights_and_importance_ratio():
    trainer_logprobs = torch.log(torch.full((3,), 0.2, device="cuda")).requires_grad_()
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.log(torch.full((3,), 0.1, device="cuda")),
        ref_logprobs=None,
        advantages=torch.tensor([-2.0, 0.0, 0.5], device="cuda"),
        loss_mask=torch.ones(3, dtype=torch.bool, device="cuda"),
    )
    rl_loss_fn = setup_rl_loss_fn(SPMALossConfig(eta=0.99, reward_range=(0.0, 1.0), dppo_mask_high=1.0, kl_tau=0.0))

    result = rl_loss_fn(inputs)
    expected_weights = torch.tensor([0.01, 1.0, 1.495], device="cuda")
    expected_ratio = torch.tensor(2.0, device="cuda")
    assert torch.isclose(result.loss, -expected_ratio * expected_weights.sum(), atol=1e-6)
    assert torch.isclose(result.metrics["spma_weight"], expected_weights.mean(), atol=1e-6)
    assert torch.isclose(result.metrics["spma_advantage_clipped"], torch.tensor(1 / 3, device="cuda"))

    result.loss.backward()
    assert torch.allclose(trainer_logprobs.grad, -expected_ratio * expected_weights, atol=1e-6)


def test_spma_dppo_mask_follows_nonnegative_policy_weight():
    inputs = LossInputs(
        trainer_logprobs=torch.log(torch.tensor([0.5], device="cuda")),
        inference_logprobs=torch.log(torch.tensor([0.1], device="cuda")),
        ref_logprobs=None,
        advantages=torch.tensor([-1.0], device="cuda"),
        loss_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
    )
    rl_loss_fn = setup_rl_loss_fn(SPMALossConfig(eta=0.99, kl_tau=0.0))

    result = rl_loss_fn(inputs)
    assert result.loss == 0
    assert result.metrics["is_masked_high"] == 1
    assert result.metrics["is_masked_low"] == 0


def test_default_loss_preserves_zero_advantage_masking_metrics():
    inputs = LossInputs(
        trainer_logprobs=torch.log(torch.tensor([0.1], device="cuda")),
        inference_logprobs=torch.log(torch.tensor([0.5], device="cuda")),
        ref_logprobs=None,
        advantages=torch.zeros(1, device="cuda"),
        loss_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
    )
    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(kl_tau=0.0))

    result = rl_loss_fn(inputs)
    assert result.loss == 0
    assert result.metrics["is_masked"] == 1
    assert result.metrics["is_masked_high"] == 0
    assert result.metrics["is_masked_low"] == 0


def test_default_loss_matches_pre_refactor_formula_and_gradient():
    trainer_logprobs = torch.log(torch.tensor([0.5, 0.1, 0.2, 0.3], device="cuda")).requires_grad_()
    inference_logprobs = torch.log(torch.tensor([0.1, 0.5, 0.2, 0.25], device="cuda"))
    advantages = torch.tensor([1.0, -1.0, 0.0, 0.5], device="cuda")
    loss_mask = torch.tensor([True, True, True, False], device="cuda")
    loss_weights = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    config = DefaultLossConfig(dppo_mask_low=0.2, dppo_mask_high=0.2, adv_tau=0.7, kl_tau=0.003)
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_weights=loss_weights,
    )

    result = setup_rl_loss_fn(config)(inputs)
    result.loss.backward()
    actual_gradient = trainer_logprobs.grad.clone()

    legacy_logprobs = trainer_logprobs.detach().clone().requires_grad_()
    log_ratio = legacy_logprobs - inference_logprobs
    ratio = torch.exp(log_ratio)
    probability_delta = torch.exp(legacy_logprobs) - torch.exp(inference_logprobs)
    positive = advantages > 0
    invalid = torch.where(
        positive, probability_delta > config.dppo_mask_high, probability_delta < -config.dppo_mask_low
    )
    keep = loss_mask & ~invalid
    expected_loss = (
        -(keep * config.adv_tau * advantages * ratio) + config.kl_tau * loss_mask * log_ratio**2
    ) * loss_weights
    expected_loss = expected_loss.sum()
    expected_loss.backward()

    assert torch.isclose(result.loss, expected_loss, atol=1e-6)
    assert torch.allclose(actual_gradient, legacy_logprobs.grad, atol=1e-6)


@pytest.mark.parametrize(
    ("loss_config", "trainer_probability", "inference_probability", "advantage", "expected_masks"),
    [
        (DefaultLossConfig(), 0.5, 0.1, 1.0, (True, True, False)),
        (DefaultLossConfig(), 0.1, 0.5, -1.0, (True, False, True)),
        (DefaultLossConfig(), 0.1, 0.5, 0.0, (True, False, False)),
        (SPMALossConfig(eta=0.99), 0.5, 0.1, -1.0, (True, True, False)),
    ],
)
def test_token_export_masks_match_loss(
    loss_config,
    trainer_probability,
    inference_probability,
    advantage,
    expected_masks,
):
    trainer_logprobs = torch.log(torch.tensor([trainer_probability], device="cuda"))
    inference_logprobs = torch.log(torch.tensor([inference_probability], device="cuda"))
    advantages = torch.tensor([advantage], device="cuda")
    loss_mask = torch.ones(1, dtype=torch.bool, device="cuda")
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
    )
    result = setup_rl_loss_fn(loss_config)(inputs)
    export = _compute_export_tensors(
        {
            "inference_logprobs": inference_logprobs,
            "advantages": advantages,
            "loss_mask": loss_mask,
        },
        trainer_logprobs,
        loss_config,
    )

    exported_masks = tuple(bool(export[name].item()) for name in ("is_masked", "is_masked_high", "is_masked_low"))
    loss_masks = tuple(bool(result.metrics[name].item()) for name in ("is_masked", "is_masked_high", "is_masked_low"))
    assert exported_masks == expected_masks
    assert loss_masks == expected_masks


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_setup_rl_loss_fn_with_custom_config():
    """Test setup_rl_loss_fn with CustomLossConfig importing a custom loss."""
    loss_config = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss._dummy_custom_loss",
        kwargs={"multiplier": 2.0},
    )
    rl_loss_fn = setup_rl_loss_fn(loss_config)

    inputs = LossInputs(
        trainer_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        inference_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        ref_logprobs=None,
        advantages=torch.randn(50).cuda(),
        loss_mask=torch.ones(50, dtype=torch.bool).cuda(),
    )

    result = rl_loss_fn(inputs)
    assert isinstance(result, LossOutputs)
    assert result.loss.shape == ()
    assert "custom_metric" in result.metrics


def test_ce_component_matches_masked_nll():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]
    rl_weights = [torch.zeros(3, dtype=torch.float32).cuda()]
    ce_weights = [torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=2,
        ref_kl_scale=1,
    )

    # loss = -sum(member logprobs) / ce_scale = -(-0.1 - 0.2) / 2 = 0.15
    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics
    assert "mismatch_kl" not in metrics


def test_ce_component_applies_weights():
    """ECHO-style observation training: the ce weight stream scales the NLL per token."""
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]
    rl_weights = [torch.zeros(3, dtype=torch.float32).cuda()]
    ce_weights = [torch.tensor([0.1, 0.0, 0.1], dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )

    # loss = 0.1 * (0.1 + 0.2) = 0.03
    assert torch.isclose(loss, torch.tensor(0.03, device=loss.device), atol=1e-6)


def test_explicit_rl_weights_match_absent_stream():
    """An explicit all-ones rl stream must equal the rl_weights=None hot path."""
    torch.manual_seed(0)
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda()]
    loss_mask = [torch.rand(50).cuda() > 0.3]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    kwargs = dict(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )
    loss_absent, _ = compute_loss(rl_weights=None, **kwargs)
    loss_explicit, _ = compute_loss(rl_weights=[torch.ones(50, dtype=torch.float32).cuda()], **kwargs)

    assert torch.equal(loss_absent, loss_explicit)


def test_disjoint_components_in_one_sequence():
    """ECHO/OPD-shaped sequence: rl, ce, and ref_kl on disjoint token sets."""
    n = 12
    torch.manual_seed(1)
    trainer_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    ref_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    advantages = [torch.randn(n).cuda()]
    loss_mask = [torch.ones(n, dtype=torch.bool).cuda()]
    rl_weights = torch.zeros(n, dtype=torch.float32)
    rl_weights[:4] = 1.0
    ce_weights = torch.zeros(n, dtype=torch.float32)
    ce_weights[4:8] = 1.0
    ref_kl_weights = torch.zeros(n, dtype=torch.float32)
    ref_kl_weights[8:] = 1.0

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=[rl_weights.cuda()],
        ce_weights=[ce_weights.cuda()],
        ref_kl_weights=[ref_kl_weights.cuda()],
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )

    assert loss.shape == ()
    assert "nll" in metrics
    assert "ref_kl" in metrics
    assert "is_masked" in metrics


def test_empty_components_keep_backward_valid():
    """A fully truncated distillation sample (stamped streams survive truncation
    as all-zero prefixes) must train as a zero-gradient no-op, not crash backward."""
    trainer_logprobs = [torch.randn(6, dtype=torch.float32, device="cuda", requires_grad=True)]
    inference_logprobs = [torch.zeros(6, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(6, dtype=torch.float32).cuda()]
    loss_mask = [torch.zeros(6, dtype=torch.bool).cuda()]
    rl_weights = [torch.zeros(6, dtype=torch.float32).cuda()]
    ce_weights = [torch.zeros(6, dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )

    assert torch.equal(loss, torch.zeros_like(loss))
    loss.backward()
    assert trainer_logprobs[0].grad is not None
    assert torch.equal(trainer_logprobs[0].grad, torch.zeros_like(trainer_logprobs[0].grad))


def test_overlapping_components_sum():
    """Components may overlap on the same token (e.g. RL + a CE behavior-cloning
    regularizer): the total is the sum of each component computed alone, each
    over its own normalization."""
    n = 8
    torch.manual_seed(2)
    trainer_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    advantages = [torch.randn(n).cuda()]
    loss_mask = [torch.ones(n, dtype=torch.bool).cuda()]
    ce_weights = [torch.full((n,), 0.5, dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    kwargs = dict(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=4,
        ce_scale=8,
        ref_kl_scale=1,
    )
    rl_only, _ = compute_loss(rl_weights=None, ce_weights=None, **kwargs)
    ce_only, _ = compute_loss(rl_weights=[torch.zeros(n, dtype=torch.float32).cuda()], ce_weights=ce_weights, **kwargs)
    both, _ = compute_loss(rl_weights=None, ce_weights=ce_weights, **kwargs)

    assert torch.isclose(both, rl_only + ce_only, atol=1e-6)


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )
