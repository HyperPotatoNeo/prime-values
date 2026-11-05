import math

import pytest
import torch

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, IPOLossConfig
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, compute_entropy, compute_loss, setup_rl_loss_fn

pytestmark = [pytest.mark.gpu]


@pytest.mark.parametrize("loss_type", ["dppo", "ipo"])
def test_policy_loss_matches_tabulated_probability_objective_and_logit_gradient(loss_type):
    # The first four actions move far in both directions for both advantage signs.
    # DPPO rejects only movement in the advantage direction; IPO rejects either.
    probabilities = [0.7, 0.1, 0.1, 0.7, 0.35, 0.25, 0.4]
    behavior = [0.3, 0.5, 0.5, 0.3, 0.3, 0.3, 0.4]
    advantages = [2.0, 2.0, -3.0, -3.0, 1.0, -1.0, 100.0]
    weights = [0.5, 1.5, 0.25, 2.0, 0.75, 1.25, 3.0]
    accepted = [False, True, False, True, True, True] if loss_type == "dppo" else [False] * 4 + [True] * 2
    adv_tau, kl_tau = 1.3, 0.07
    config = (
        DefaultLossConfig(dppo_mask_low=0.2, dppo_mask_high=0.2, adv_tau=adv_tau, kl_tau=kl_tau)
        if loss_type == "dppo"
        else IPOLossConfig(ipo_threshold=0.2, adv_tau=adv_tau, kl_tau=kl_tau)
    )
    logits = torch.tensor(
        [[math.log(p), math.log1p(-p)] for p in probabilities],
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    inputs = LossInputs(
        trainer_logprobs=logits.log_softmax(-1)[:, 0],
        inference_logprobs=torch.tensor(behavior, dtype=torch.float64, device="cuda").log(),
        ref_logprobs=None,
        advantages=torch.tensor(advantages, dtype=torch.float64, device="cuda"),
        loss_mask=torch.tensor([True] * 6 + [False], device="cuda"),
        loss_weights=torch.tensor(weights, dtype=torch.float64, device="cuda"),
    )
    result = setup_rl_loss_fn(config)(inputs)
    result.loss.backward()

    expected_loss = 0.0
    expected_gradient = []
    for p, q, advantage, weight, keep in zip(
        probabilities[:6], behavior[:6], advantages[:6], weights[:6], accepted, strict=True
    ):
        ratio = p / q
        drift = math.log(ratio)
        policy_term = -adv_tau * advantage * ratio if keep else 0.0
        expected_loss += weight * (policy_term + kl_tau * drift**2)
        # d log softmax(action) / d logits = one_hot(action) - probabilities.
        coefficient = weight * (policy_term + 2 * kl_tau * drift)
        expected_gradient.append([coefficient * (1 - p), -coefficient * (1 - p)])
    expected_gradient.append([0.0, 0.0])
    torch.testing.assert_close(result.loss, result.loss.new_tensor(expected_loss), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(logits.grad, logits.new_tensor(expected_gradient), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("partition", [(9,), (2, 3, 4), (1,) * 9])
def test_mixed_component_gradient_is_invariant_to_sequence_partition(partition):
    # CE/ref-KL may train observation tokens outside the policy action mask.
    # Global component counts are supplied by the trainer, not inferred here.
    p = [0.35, 0.25, 0.4, 0.6, 0.2, 0.4, 0.1, 0.7, 0.3]
    q = [0.3, 0.3, 0.4, 0.5, 0.3, 0.35, 0.5, 0.7, 0.3]
    reference = [0.3, 0.3, 0.4, 0.5, 0.3, 0.55, 0.6, 0.7, 0.3]
    adv = [2.0, -3.0, 100.0, 0.0, 0.0, 999.0, -999.0, 0.0, 0.0]
    rl_weight = [0.5, 1.25, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ce_weight = [0.0, 0.25, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0]
    ref_weight = [0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 1.25, 0.0, 0.0]
    logits = torch.tensor(
        [[math.log(prob), math.log1p(-prob)] for prob in p],
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )

    def split(values, dtype=torch.float64):
        return list(torch.tensor(values, dtype=dtype, device="cuda").split(partition))

    loss, _ = compute_loss(
        trainer_logprobs=list(logits.log_softmax(-1)[:, 0].split(partition)),
        inference_logprobs=split([math.log(prob) for prob in q]),
        ref_logprobs=split([math.log(prob) for prob in reference]),
        advantages=split(adv),
        loss_mask=split([True, True] + [False] * 7, torch.bool),
        rl_weights=split(rl_weight),
        ce_weights=split(ce_weight),
        ref_kl_weights=split(ref_weight),
        rl_loss_fn=setup_rl_loss_fn(DefaultLossConfig()),
        rl_scale=2,
        ce_scale=3,
        ref_kl_scale=2,
    )
    loss.backward()

    objective = 0.0
    coefficients = [0.0] * 9
    for index in (0, 1):
        ratio = p[index] / q[index]
        drift = math.log(ratio)
        weight = rl_weight[index] / 2
        objective += weight * (-adv[index] * ratio + 0.001 * drift**2)
        coefficients[index] += weight * (-adv[index] * ratio + 0.002 * drift)
    for index in (1, 3, 4):
        weight = ce_weight[index] / 3
        objective -= weight * math.log(p[index])
        coefficients[index] -= weight
    for index, accepted in ((5, True), (6, False)):
        ratio = p[index] / q[index]
        drift = math.log(ratio)
        weight = ref_weight[index] / 2
        # The reference-derived advantage is fixed during the policy update.
        policy_term = -math.log(reference[index] / p[index]) * ratio if accepted else 0.0
        objective += weight * (policy_term + 0.001 * drift**2)
        coefficients[index] += weight * (policy_term + 0.002 * drift)
    gradient = [
        [coefficient * (1 - prob), -coefficient * (1 - prob)] for prob, coefficient in zip(p, coefficients, strict=True)
    ]
    torch.testing.assert_close(loss, loss.new_tensor(objective), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(logits.grad, logits.new_tensor(gradient), rtol=1e-10, atol=1e-10)


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
