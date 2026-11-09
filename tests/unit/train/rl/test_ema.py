import copy
import math

import pytest
import torch
import torch.distributed.checkpoint as dcp
from pydantic import ValidationError
from torch import nn

from prime_rl.configs.trainer import EMAKLConfig, ReinforceLossConfig, TrainerConfig
from prime_rl.trainer.rl.ema import EMAReference
from prime_rl.trainer.rl.loss import LossInputs, reverse_kl_under_sampler, setup_rl_loss_fn


@pytest.mark.parametrize("logits", [[-0.2, 1.7, -2.0], [-1000.0, 0.0, -30.0]])
def test_reverse_kl_matches_exact_off_policy_expectation_and_gradient(logits):
    logits = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    sampler = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64).log().requires_grad_()
    reference = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64).log().requires_grad_()
    policy = logits.log_softmax(-1)
    estimate = (sampler.detach().exp() * reverse_kl_under_sampler(policy, reference, sampler)).sum()
    exact = (policy.exp() * (policy - reference.detach())).sum()
    gradient = torch.autograd.grad(estimate, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(exact, logits)[0]
    torch.testing.assert_close(estimate, exact, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert torch.autograd.grad(estimate, (sampler, reference), allow_unused=True) == (None, None)


@pytest.mark.parametrize("cap", [None, 0.5, 2.0])
@pytest.mark.parametrize("coefficient", [0.0, 0.03])
def test_reinforce_matches_matched_objective_with_masks_and_weights(cap, coefficient):
    policy = torch.tensor([-0.1, -2.4, -0.7, float("nan")], dtype=torch.float64, requires_grad=True)
    sampler = torch.tensor([-4.0, -0.2, -2.0, float("nan")], dtype=torch.float64)
    reference = torch.tensor([-0.8, -0.4, -1.4, float("nan")], dtype=torch.float64)
    advantages = torch.tensor([2.0, -3.0, 0.0, float("nan")], dtype=torch.float64)
    weights = torch.tensor([0.5, 1.25, 2.0, float("nan")], dtype=torch.float64)
    config = ReinforceLossConfig(importance_cap=cap, ema_kl=EMAKLConfig(coefficient=coefficient))
    result = setup_rl_loss_fn(config)(
        LossInputs(
            policy, sampler, reference if coefficient else None, advantages, torch.tensor([1, 1, 1, 0]).bool(), weights
        )
    )
    p, q, e, advantage = policy[:3], sampler[:3], reference[:3], advantages[:3]
    importance = torch.ones_like(p) if cap is None else torch.exp(torch.clamp_max(p.detach() - q, math.log(cap)))
    expected_tokens = -advantage * importance * p
    if coefficient:
        # The policy-gradient ratio is capped; the differentiable KL ratio is not.
        expected_tokens = expected_tokens + coefficient * (p - q).exp() * ((e - p).exp() - (e - p) - 1)
    expected = (expected_tokens * weights[:3]).sum()
    actual_grad = torch.autograd.grad(result.loss, policy, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, policy)[0]
    torch.testing.assert_close(result.loss, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
    assert actual_grad[-1] == 0
    assert bool(actual_grad[2] != 0) == bool(coefficient)
    empty = setup_rl_loss_fn(config)(
        LossInputs(policy, sampler, reference, advantages, torch.zeros(4, dtype=torch.bool), weights)
    )
    assert empty.loss == 0
    assert torch.isfinite(torch.autograd.grad(empty.loss, policy)[0]).all()


@pytest.mark.parametrize("decay", [0.0, 0.95])
def test_ema_tracks_bf16_publications_and_resumes_exactly(tmp_path, decay):
    policy = nn.Linear(3, 2, bias=False)
    policy.register_buffer("counter", torch.tensor(0))
    ema = EMAReference(copy.deepcopy(policy), policy, decay)
    expected = policy.weight.detach().to(torch.bfloat16).float()
    torch.testing.assert_close(ema.model.weight, expected, rtol=0, atol=0)
    assert not ema.model.training and not ema.model.weight.requires_grad
    assert (ema.updates, ema.last_publication) == (0, -1)
    with torch.no_grad():
        policy.weight.add_(0.0103)
        policy.counter.add_(1)
    ema.update(policy, 0)
    expected.mul_(decay).add_(policy.weight.detach().to(torch.bfloat16).float(), alpha=1 - decay)
    torch.testing.assert_close(ema.model.weight, expected, rtol=0, atol=0)
    assert ema.model.counter == 1
    with pytest.raises(ValueError, match="strictly increasing"):
        ema.update(policy, 0)

    dcp.save({"ema_reference": ema}, checkpoint_id=tmp_path / "checkpoint")
    resumed = EMAReference(copy.deepcopy(policy), policy, decay)
    dcp.load({"ema_reference": resumed}, checkpoint_id=tmp_path / "checkpoint")
    assert (resumed.updates, resumed.last_publication) == (1, 0)
    torch.testing.assert_close(resumed.model.weight, ema.model.weight, rtol=0, atol=0)
    with torch.no_grad():
        policy.weight.add_(0.0271)
    ema.update(policy, 3)
    resumed.update(policy, 3)
    torch.testing.assert_close(resumed.model.weight, ema.model.weight, rtol=0, atol=0)
    assert (resumed.updates, resumed.last_publication) == (2, 3)
    incompatible = EMAReference(copy.deepcopy(policy), policy, 0.5)
    with pytest.raises(ValueError, match="decay"):
        incompatible.load_state_dict(ema.state_dict())


def test_ema_config_is_opt_in_and_roundtrips():
    assert TrainerConfig().loss.type == "default"
    assert ReinforceLossConfig().ema_kl is None
    assert EMAKLConfig().coefficient == 0
    assert EMAKLConfig().decay == 0.95
    disabled = TrainerConfig(model={"lora": {}}, loss={"type": "reinforce", "ema_kl": {"coefficient": 0}})
    assert disabled.loss.ema_kl.coefficient == 0
    enabled = TrainerConfig(
        model={"name": "Qwen/Qwen3-4B", "impl": "custom"},
        loss={"type": "reinforce", "ema_kl": {"coefficient": 0.01}},
    )
    assert enabled.loss.ema_kl.decay == 0.95
    assert TrainerConfig.model_validate_json(enabled.model_dump_json()) == enabled


@pytest.mark.parametrize("field", ["coefficient", "decay"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_ema_config_rejects_invalid_numbers(field, value):
    with pytest.raises(ValidationError):
        EMAKLConfig(**{field: value})


@pytest.mark.parametrize(
    "override",
    [
        {"max_concurrent_runs": 2},
        {"model": {"cp": 2}},
        {"model": {"lora": {}}},
        {"model": {"impl": "hf"}},
        {"model": {"name": "Qwen/Qwen3-0.6B"}},
        {"model": {"fused_lm_head_token_chunk_size": "disabled"}},
        {"model": {"fp8": True}},
        {"enable_router_replay": True},
        {"weight_broadcast": {"type": "nccl", "quantize_in_weight_transfer": True}},
    ],
)
def test_ema_config_rejects_unsupported_runtime(override):
    config = {
        "model": {"name": "Qwen/Qwen3-4B", "impl": "custom", **override.get("model", {})},
        "loss": {"type": "reinforce", "ema_kl": {"coefficient": 0.01}},
        **{key: value for key, value in override.items() if key != "model"},
    }
    with pytest.raises(ValidationError, match="EMA KL"):
        TrainerConfig(**config)
