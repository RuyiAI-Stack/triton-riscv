import math

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


def make_flash_inputs(batch=1, heads=2, seqlen=10, head_dim=16):
    torch.manual_seed(1)
    q = torch.randn((batch, heads, seqlen, head_dim), device="cpu", dtype=torch.float32)
    k = torch.randn((batch, heads, seqlen, head_dim), device="cpu", dtype=torch.float32)
    v = torch.randn((batch, heads, seqlen, head_dim), device="cpu", dtype=torch.float32)
    return q.contiguous(), k.contiguous(), v.contiguous()


def attention_flash_reference(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def _validate_attention_flash_inputs(q, k, v):
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
def _flash_row_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seqlen,
    head_dim,
    stride_qs,
    stride_qd,
    stride_ks,
    stride_kd,
    stride_vs,
    stride_vd,
    stride_os,
    stride_od,
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + row * stride_qs + offs_d * stride_qd,
        mask=offs_d < head_dim,
        other=0.0,
    ).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start in range(0, seqlen, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        k = tl.load(
            k_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=(offs_n[:, None] < seqlen) & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=(offs_n[:, None] < seqlen) & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(offs_n < seqlen, scores, -float("inf"))

        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + row * stride_os + offs_d * stride_od, out, mask=offs_d < head_dim)


def attention_flash_triton(q, k, v):
    _validate_attention_flash_inputs(q, k, v)

    batch, heads, seqlen, head_dim = q.shape
    block_d = triton.next_power_of_2(head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)

    for b in range(batch):
        for h in range(heads):
            q_slice = q[b, h]
            k_slice = k[b, h]
            v_slice = v[b, h]
            out_slice = torch.empty((seqlen, head_dim), device="cpu", dtype=torch.float32)

            _flash_row_kernel[(seqlen,)](
                q_slice,
                k_slice,
                v_slice,
                out_slice,
                seqlen,
                head_dim,
                q_slice.stride(0),
                q_slice.stride(1),
                k_slice.stride(0),
                k_slice.stride(1),
                v_slice.stride(0),
                v_slice.stride(1),
                out_slice.stride(0),
                out_slice.stride(1),
                scale,
                BLOCK_N=4,
                BLOCK_D=block_d,
            )

            out[b, h].copy_(out_slice)

    return out


def test_attention_flash_matches_torch(device):
    q, k, v = make_flash_inputs(batch=1, heads=2, seqlen=10, head_dim=16)
    out = attention_flash_triton(q, k, v)
    ref = attention_flash_reference(q, k, v)
    torch.testing.assert_close(out, ref, atol=5e-4, rtol=5e-4)

    with pytest.raises(TypeError):
        attention_flash_triton(q.to(torch.float16), k, v)
    with pytest.raises(ValueError):
        attention_flash_triton(q, k, v[:, :, :, :-1])


def test_attention_flash_tail_block(device):
    q, k, v = make_flash_inputs(batch=1, heads=1, seqlen=13, head_dim=24)
    out = attention_flash_triton(q, k, v)
    ref = attention_flash_reference(q, k, v)
    torch.testing.assert_close(out, ref, atol=5e-4, rtol=5e-4)
