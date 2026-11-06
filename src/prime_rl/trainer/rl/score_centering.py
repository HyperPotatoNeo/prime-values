"""Marek & Ryabinin (2026), Eq. 14: top-k head with a proportional tail.

The candidate log probabilities retain full-vocabulary normalization. The
sampler's selected action need not occur in its top-k head.
"""

import math

import torch
from torch import Tensor


def importance_weight(log_ratio: Tensor, cap: float | None) -> Tensor:
    if cap is None:
        return torch.ones_like(log_ratio)
    return torch.exp(torch.clamp_max(log_ratio, math.log(cap)))


def centering_term(policy_head: Tensor, sampler_head: Tensor, cap: float | None, eps: float):
    # All coefficient dependence on the current policy is stopped (Eq. 14).
    with torch.no_grad():
        p = policy_head.exp()
        q = sampler_head.exp()
        p_tail = (1 - p.sum(-1)).clamp_min(eps)
        q_tail = (1 - q.sum(-1)).clamp_min(eps)
        rho = q_tail / p_tail
        alpha = rho * importance_weight(-rho.log(), cap)
        coefficients = q * importance_weight(policy_head - sampler_head, cap) - alpha.unsqueeze(-1) * p
    return (coefficients * policy_head).sum(-1), q_tail, p_tail


def reverse_kl_under_sampler(policy_logp: Tensor, reference_logp: Tensor, sampler_logp: Tensor) -> Tensor:
    """Unclipped importance-corrected k3; expectation under q equals KL(p || e).

    q and e are fixed. Differentiating this expression gives the exact
    reverse-KL gradient in expectation, including when q != p. Clipping the
    importance ratio here would change the regularizer.
    """
    log_ratio = policy_logp - sampler_logp.detach()
    ref_delta = reference_logp.detach() - policy_logp
    ratio = torch.exp(log_ratio)
    stable_near = ratio * (torch.expm1(ref_delta.clamp_max(20)) - ref_delta)
    stable_far = torch.exp(reference_logp.detach() - sampler_logp.detach()) - ratio * (ref_delta + 1)
    return torch.where(ref_delta > 20, stable_far, stable_near)
