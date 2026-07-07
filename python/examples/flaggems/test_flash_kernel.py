import pytest
import torch
import triton

from .flash_api import mha_fwd
from .flash_kernel import (
    flash_fwd_kernel,
    flash_varlen_fwd_kernel,
)


@pytest.mark.parametrize("B, H, Q, K, D", [(1, 2, 16, 16, 32)])
def test_flash_kernel_compile_small(B, H, Q, K, D):
    """Compile test for flash_fwd_kernel with minimal shape."""
    q = torch.randn(B, H, Q, D, dtype=torch.float16)
    k = torch.randn(B, H, Q, D, dtype=torch.float16)
    v = torch.randn(B, H, Q, D, dtype=torch.float16)
    out = torch.empty_like(q)
    lse = torch.empty((B, H, Q), dtype=torch.float32)

    if q.device.type == "cpu":
        out_ref, *_ = mha_fwd(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            out=None,
            alibi_slopes=None,
            p_dropout=0.0,
            softmax_scale=1.0 / (D**0.5),
            is_causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            return_softmax=False,
        )
        assert out_ref.shape == (B, Q, H, D)
        return

    def grid(args):
        return (triton.cdiv(Q, args["BLOCK_M"]), B * H)

    flash_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        torch.empty((), device=q.device),  # p_ptr
        lse,  # softmax_lse_ptr
        q.stride(-3),
        k.stride(-3),
        v.stride(-3),
        q.stride(-2),
        k.stride(-2),
        v.stride(-2),
        out.stride(-3),
        out.stride(-2),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        out.stride(0),
        False,
        None,
        False,
        None,
        False,
        None,
        B,
        0,
        H,
        H,
        1,
        Q,
        K,
        Q,
        K,
        D,
        32,
        False,
        0.0,
        1.0,
        1.442695,
        False,
        0.0,
        1.0,
        0,
        torch.empty(2, dtype=torch.int64),
        False,
        False,
        False,
        -1,
        -1,
        False,
        False,
        False,
        None,
        0,
        0,
        None,
        0,
        0,
        IS_EVEN_MN=True,
        PRE_LOAD_V=False,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
        num_warps=4,
        num_stages=1,
    )
    assert out.shape == q.shape


@pytest.mark.parametrize("B, H, total_q, K, D", [(1, 2, 32, 32, 32)])
def test_varlen_fwd_kernel_compile_small(B, H, total_q, K, D):
    """Compile test for flash_varlen_fwd_kernel."""
    q = torch.randn(total_q, H, D, dtype=torch.float16)
    k = torch.randn(total_q, H, D, dtype=torch.float16)
    v = torch.randn(total_q, H, D, dtype=torch.float16)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32) * (total_q // (B + 1))
    cu_seqlens_q[-1] = total_q
    cu_seqlens_k = cu_seqlens_q.clone()
    lse = torch.empty((H, total_q), dtype=torch.float32)
    max_seqlen_q = total_q // (B + 1)
    max_seqlen_k = max_seqlen_q

    seqlen_q_rounded = ((max_seqlen_q + 127) // 128) * 128
    seqlen_k_rounded = ((max_seqlen_k + 31) // 32) * 32

    if q.device.type == "cpu":
        q_batched = q.view(B, total_q // B, H, D)
        k_batched = k.view(B, total_q // B, H, D)
        v_batched = v.view(B, total_q // B, H, D)
        out_ref, *_ = mha_fwd(
            q_batched,
            k_batched,
            v_batched,
            out=None,
            alibi_slopes=None,
            p_dropout=0.0,
            softmax_scale=1.0 / (D**0.5),
            is_causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            return_softmax=False,
        )
        assert out_ref.shape == q_batched.shape
        return

    def grid(args):
        return (
            triton.cdiv(max_seqlen_q, args["BLOCK_M"]),
            B,
            H,
        )

    flash_varlen_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        torch.empty((), device=q.device),
        lse,
        q.stride(-3),
        k.stride(-3),
        v.stride(-3),
        q.stride(-2),
        k.stride(-2),
        v.stride(-2),
        out.stride(-3),
        out.stride(-2),
        0,
        0,
        0,
        0,
        True,
        cu_seqlens_q,
        True,
        cu_seqlens_k,
        False,
        None,
        B,
        0,
        H,
        H,
        1,
        max_seqlen_q,
        max_seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        D,
        32,
        False,
        0.0,
        1.0 / (D**0.5),
        1.442695 * (1.0 / (D**0.5)),
        False,
        0.0,
        1.0,
        0,
        torch.empty(2, dtype=torch.int64),
        False,
        False,
        False,
        -1,
        -1,
        False,
        False,
        False,
        None,
        0,
        0,
        None,
        0,
        0,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=1,
    )
    assert out.shape == q.shape
