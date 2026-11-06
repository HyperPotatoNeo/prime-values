"""Opaque FA2/FA3 kernels for checkpointed, compiled shared attention.

The public autograd.Function wrappers graph-break on aliased q/k sequence
boundaries. Opaque forward/backward ops preserve the existing kernel settings
without exposing those Python wrappers to Dynamo. This boundary covers the
shared attention call, not the independent context-parallel implementations.
"""

import functools

import torch

try:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward, _flash_attn_varlen_forward
except ImportError:
    _flash_attn_varlen_backward = _flash_attn_varlen_forward = None

try:
    from flash_attn_interface import _flash_attn_backward as _flash_attn_3_bwd
    from flash_attn_interface import _flash_attn_forward as _flash_attn_3_fwd
except ImportError:
    _flash_attn_3_bwd = _flash_attn_3_fwd = None


@torch.library.custom_op("prime_rl_attn::flash_attn_3_varlen", mutates_args=())
def _flash_attn_3_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _ = _flash_attn_3_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    return out, lse


@_flash_attn_3_varlen.register_fake
def _(q, k, v, *args):
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))
    lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("prime_rl_attn::flash_attn_3_varlen_backward", mutates_args=())
def _flash_attn_3_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_3_bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dq=dq,
        dk=dk,
        dv=dv,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        deterministic=deterministic,
    )
    return dq, dk, dv


@_flash_attn_3_varlen_backward.register_fake
def _(dout, q, k, v, *args):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# FA2's public flash_attn_varlen_func pads the head dim to a multiple of 8 around its kernels; these ops call
# the kernels directly, so they pad the same way.
def pad_head_dim_to_multiple_of_8_for_fa2(t: torch.Tensor) -> torch.Tensor:
    head_dim = t.shape[-1]
    if head_dim % 8 == 0:
        return t
    return torch.nn.functional.pad(t, [0, 8 - head_dim % 8])


@torch.library.custom_op("prime_rl_attn::flash_attn_2_varlen", mutates_args=())
def _flash_attn_2_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    q, k, v = (pad_head_dim_to_multiple_of_8_for_fa2(t) for t in (q, k, v))
    out_padded, lse, _, _ = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    return out_padded[..., :head_dim], lse


# FA2's kernel allocates its outputs with empty_like on the padded inputs, so the fakes keep their strides and padding.
@_flash_attn_2_varlen.register_fake
def _(q, k, v, *args):
    head_dim = q.shape[-1]
    out = torch.empty_like(pad_head_dim_to_multiple_of_8_for_fa2(q))[..., :head_dim]
    lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("prime_rl_attn::flash_attn_2_varlen_backward", mutates_args=())
def _flash_attn_2_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    dout, q, k, v, out = (pad_head_dim_to_multiple_of_8_for_fa2(t) for t in (dout, q, k, v, out))
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        0.0,
        None,
        deterministic,
    )
    return dq[..., :head_dim], dk[..., :head_dim], dv[..., :head_dim]


@_flash_attn_2_varlen_backward.register_fake
def _(dout, q, k, v, *args):
    head_dim = q.shape[-1]
    return tuple(torch.empty_like(pad_head_dim_to_multiple_of_8_for_fa2(t))[..., :head_dim] for t in (q, k, v))


def _flash_attn_varlen_setup_context(ctx, inputs, output) -> None:
    q, k, v, cu_seqlens_q, cu_seqlens_k, *scalar_args = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
    ctx.scalar_args = scalar_args


def _flash_attn_varlen_autograd(backward_op, ctx, dout: torch.Tensor, _dlse: torch.Tensor | None):
    dq, dk, dv = backward_op(dout.contiguous(), *ctx.saved_tensors, *ctx.scalar_args)
    return dq, dk, dv, None, None, None, None, None, None, None, None, None


_flash_attn_2_varlen.register_autograd(
    functools.partial(_flash_attn_varlen_autograd, _flash_attn_2_varlen_backward),
    setup_context=_flash_attn_varlen_setup_context,
)
_flash_attn_3_varlen.register_autograd(
    functools.partial(_flash_attn_varlen_autograd, _flash_attn_3_varlen_backward),
    setup_context=_flash_attn_varlen_setup_context,
)


def flash_attn_2_varlen_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
) -> torch.Tensor:
    """FA2 for shared attention: zero dropout, no ALiBi or softcap."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    out, _ = _flash_attn_2_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        *window_size,
        deterministic,
    )
    return out


def flash_attn_3_varlen_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
) -> torch.Tensor:
    """FA3 for shared attention: no ALiBi, softcap, or FP8 inputs."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    out, _ = _flash_attn_3_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        *window_size,
        deterministic,
    )
    return out
