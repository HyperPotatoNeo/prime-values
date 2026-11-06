import copy

import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn
from prime_rl.trainer.rl.metrics import accumulate_token_metrics, prepare_token_env_ids
from prime_rl.trainer.utils import TensorSummaries


def test_token_instrumentation_preserves_policy_outputs_full_gradients_and_update():
    torch.manual_seed(61)
    model = torch.nn.Linear(3, 5, dtype=torch.float64)
    reference = copy.deepcopy(model)
    inputs = torch.arange(18, dtype=torch.float64).reshape(6, 3).sin()
    mask = torch.tensor([[False, True, True, False, True, True]])
    labels = torch.tensor([[0, 1, 2, 3, 4, 0]])
    gradients, updates = [], []
    for current, instrument in ((reference, False), (model, True)):
        logits = current(inputs).unsqueeze(0)
        logprobs = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        inference = logprobs.detach().clone() + 0.01
        entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
        before = logprobs.detach().clone()
        loss, _ = compute_loss(
            trainer_logprobs=[logprobs[0]],
            inference_logprobs=[inference[0]],
            ref_logprobs=None,
            advantages=[torch.tensor([0, 0.5, -0.2, 0, -0.7, 0.3], dtype=torch.float64)],
            loss_mask=[mask[0]],
            rl_weights=None,
            ce_weights=None,
            ref_kl_weights=None,
            rl_loss_fn=setup_rl_loss_fn(DefaultLossConfig()),
            rl_scale=4,
            ce_scale=1,
            ref_kl_scale=1,
        )
        if instrument:
            names, ids = prepare_token_env_ids(["a", "a", "b", "b", "a", "b"], torch.device("cpu"))
            stats, diagnostics = TensorSummaries("cpu", include_std=True), TensorSummaries("cpu")
            accumulate_token_metrics(
                stats,
                diagnostics,
                entropy=entropy,
                trainer_logprobs=logprobs,
                inference_logprobs=inference,
                loss_mask=mask,
                rl_weights=None,
                ref_kl_weights=None,
                env_names=names,
                env_ids=ids,
            )
            assert all(not row.requires_grad for row in (*stats.summaries.values(), *diagnostics.summaries.values()))
            assert all(row.numel() == 5 for row in stats.summaries.values())
            torch.testing.assert_close(logprobs, before, rtol=0, atol=0)
        loss.backward()
        gradients.append({name: parameter.grad.clone() for name, parameter in current.named_parameters()})
        torch.optim.AdamW(current.parameters(), lr=0.01).step()
        updates.append({name: parameter.detach().clone() for name, parameter in current.named_parameters()})
    for name in gradients[0]:
        assert gradients[0][name].norm() > 0
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)
        torch.testing.assert_close(updates[0][name], updates[1][name], rtol=0, atol=0)


def test_token_masks_exclude_nonfinite_frozen_and_padding_logprobs():
    stats, diagnostics = TensorSummaries("cpu", include_std=True), TensorSummaries("cpu")
    names, ids = prepare_token_env_ids(["rl", "ce", "ref", ""], torch.device("cpu"))
    mismatch, mask = accumulate_token_metrics(
        stats,
        diagnostics,
        entropy=torch.tensor([[1.0, 2.0, 3.0, float("nan")]]),
        trainer_logprobs=torch.tensor([[0.0, float("nan"), 0.0, float("inf")]]),
        inference_logprobs=torch.tensor([[0.0, float("inf"), 0.0, float("nan")]]),
        loss_mask=torch.tensor([[True, True, True, False]]),
        rl_weights=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ref_kl_weights=torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        env_names=names,
        env_ids=ids,
    )
    assert mask.tolist() == [[True, False, True, False]]
    assert torch.isfinite(mismatch).all()
    torch.testing.assert_close(stats.summaries["mismatch_kl/all"], torch.tensor([2, 0, 0, 0, 0], dtype=torch.float64))
    assert stats.summaries["entropy/all"][0] == 3
    assert stats.summaries["mismatch_kl/ce"][0] == 0
    assert diagnostics.summaries["mismatch_kl/all/saturated"][0] == 2
