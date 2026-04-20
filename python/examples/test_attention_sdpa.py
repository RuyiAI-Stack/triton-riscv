import math

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


def make_sdpa_inputs(batch=1, heads=2, seqlen=8, head_dim=16):
    torch.manual_seed(0)
    q = torch.randn((batch, heads, seqlen, head_dim), device="cpu", dtype=torch.float32)
    k = torch.randn((batch, heads, seqlen, head_dim), device="cpu", dtype=torch.float32)
    v = torch.randn((batch, heads, seqlen, head_dim), device="cpu", dtype=torch.float32)
    return q.contiguous(), k.contiguous(), v.contiguous()


def attention_sdpa_reference(q, k, v, causal=False):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        mask = torch.triu(
            torch.ones((q.shape[-2], q.shape[-2]), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def _validate_attention_sdpa_inputs(q, k, v):
    tensors = {"q": q, "k": k, "v": v}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same shape")


@triton.jit
def _score_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    seqlen,
    head_dim,
    stride_qs,
    stride_qd,
    stride_ks,
    stride_kd,
    stride_ss,
    stride_sk,
    scale,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d_start in range(0, head_dim, BLOCK_D):
        d = d_start + offs_d
        q = tl.load(
            q_ptr + offs_m[:, None] * stride_qs + d[None, :] * stride_qd,
            mask=(offs_m[:, None] < seqlen) & (d[None, :] < head_dim),
            other=0.0,
        )
        k = tl.load(
            k_ptr + offs_n[:, None] * stride_ks + d[None, :] * stride_kd,
            mask=(offs_n[:, None] < seqlen) & (d[None, :] < head_dim),
            other=0.0,
        )
        acc += tl.dot(q, tl.trans(k))

    acc = acc * scale
    if causal:
        acc = tl.where(offs_n[None, :] > offs_m[:, None], -float("inf"), acc)

    tl.store(
        scores_ptr + offs_m[:, None] * stride_ss + offs_n[None, :] * stride_sk,
        acc,
        mask=(offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen),
    )


@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


@triton.jit
def _value_kernel(
    attn_ptr,
    v_ptr,
    out_ptr,
    seqlen,
    head_dim,
    stride_as,
    stride_ak,
    stride_vs,
    stride_vd,
    stride_os,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, seqlen, BLOCK_K):
        k = k_start + offs_k
        attn = tl.load(
            attn_ptr + offs_m[:, None] * stride_as + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < seqlen) & (k[None, :] < seqlen),
            other=0.0,
        )
        values = tl.load(
            v_ptr + k[:, None] * stride_vs + offs_n[None, :] * stride_vd,
            mask=(k[:, None] < seqlen) & (offs_n[None, :] < head_dim),
            other=0.0,
        )
        acc += tl.dot(attn, values)

    tl.store(
        out_ptr + offs_m[:, None] * stride_os + offs_n[None, :] * stride_od,
        acc,
        mask=(offs_m[:, None] < seqlen) & (offs_n[None, :] < head_dim),
    )


def _softmax_2d(x):
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return y


def attention_sdpa_triton(q, k, v, causal=False):
    _validate_attention_sdpa_inputs(q, k, v)

    batch, heads, seqlen, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)

    for b in range(batch):
        for h in range(heads):
            q_slice = q[b, h]
            k_slice = k[b, h]
            v_slice = v[b, h]
            scores = torch.empty((seqlen, seqlen), device="cpu", dtype=torch.float32)
            attn = torch.empty_like(scores)
            out_slice = torch.empty((seqlen, head_dim), device="cpu", dtype=torch.float32)

            score_grid = (
                triton.cdiv(seqlen, 8),
                triton.cdiv(seqlen, 8),
            )
            _score_kernel[score_grid](
                q_slice,
                k_slice,
                scores,
                seqlen,
                head_dim,
                q_slice.stride(0),
                q_slice.stride(1),
                k_slice.stride(0),
                k_slice.stride(1),
                scores.stride(0),
                scores.stride(1),
                scale,
                causal=causal,
                BLOCK_M=8,
                BLOCK_N=8,
                BLOCK_D=8,
            )

            attn.copy_(_softmax_2d(scores))

            value_grid = (
                triton.cdiv(seqlen, 8),
                triton.cdiv(head_dim, 8),
            )
            _value_kernel[value_grid](
                attn,
                v_slice,
                out_slice,
                seqlen,
                head_dim,
                attn.stride(0),
                attn.stride(1),
                v_slice.stride(0),
                v_slice.stride(1),
                out_slice.stride(0),
                out_slice.stride(1),
                BLOCK_M=8,
                BLOCK_N=8,
                BLOCK_K=8,
            )

            out[b, h].copy_(out_slice)

    return out


@pytest.mark.parametrize("causal", [False, True])
def test_attention_sdpa_matches_torch(device, causal):
    q, k, v = make_sdpa_inputs(batch=1, heads=2, seqlen=8, head_dim=16)
    out = attention_sdpa_triton(q, k, v, causal=causal)
    ref = attention_sdpa_reference(q, k, v, causal=causal)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_attention_sdpa_tail_shape(device):
    q, k, v = make_sdpa_inputs(batch=1, heads=1, seqlen=12, head_dim=16)
    out = attention_sdpa_triton(q, k, v, causal=False)
    ref = attention_sdpa_reference(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("seqlen", [11, 13])
def test_attention_sdpa_causal_tail_sequence(device, seqlen):
    q, k, v = make_sdpa_inputs(batch=1, heads=1, seqlen=seqlen, head_dim=16)
    out = attention_sdpa_triton(q, k, v, causal=True)
    ref = attention_sdpa_reference(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("head_dim", [17, 19])
def test_attention_sdpa_head_dim_tail(device, head_dim):
    q, k, v = make_sdpa_inputs(batch=1, heads=1, seqlen=8, head_dim=head_dim)
    out = attention_sdpa_triton(q, k, v, causal=False)
    ref = attention_sdpa_reference(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
