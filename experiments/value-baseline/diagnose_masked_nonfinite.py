"""Report a known safety gap without treating defective output as a contract."""

import json

import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.rl.loss import LossInputs, setup_rl_loss_fn

results = []
for excluded in (float("nan"), float("inf"), -float("inf")):
    logprobs = torch.tensor([-1.0, excluded], device="cuda", requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=logprobs,
        inference_logprobs=torch.tensor([-1.0, -1.0], device="cuda"),
        ref_logprobs=None,
        advantages=torch.tensor([1.0, 0.0], device="cuda"),
        loss_mask=torch.tensor([True, False], device="cuda"),
    )
    loss = setup_rl_loss_fn(DefaultLossConfig())(inputs).loss
    loss.backward()
    results.append(
        {
            "excluded_logprob": str(excluded),
            "finite_loss": bool(torch.isfinite(loss)),
            "finite_all_gradients": bool(torch.isfinite(logprobs.grad).all()),
            "finite_eligible_gradient": bool(torch.isfinite(logprobs.grad[0])),
            "desired_behavior": "finite loss and gradients; excluded token has zero gradient",
        }
    )
print(json.dumps(results, indent=2))
