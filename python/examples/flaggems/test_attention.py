import pytest
import torch
import torch.nn.functional as F

from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    flash_attn_varlen_opt_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_forward,
)


# scaled_dot_product_attention tests


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("nheads", [2, 4])
@pytest.mark.parametrize("seqlen", [8, 16])
@pytest.mark.parametrize("headdim", [16, 32])
def test_scaled_dot_product_attention(batch, nheads, seqlen, headdim):
    torch.manual_seed(0)
    q = torch.randn(batch, nheads, seqlen, headdim, device="cpu", dtype=torch.float32)
    k = torch.randn(batch, nheads, seqlen, headdim, device="cpu", dtype=torch.float32)
    v = torch.randn(batch, nheads, seqlen, headdim, device="cpu", dtype=torch.float32)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    tri = scaled_dot_product_attention(q, k, v, is_causal=False)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("nheads", [2, 4])
@pytest.mark.parametrize("seqlen", [8, 16])
@pytest.mark.parametrize("headdim", [16, 32])
def test_scaled_dot_product_attention_causal(batch, nheads, seqlen, headdim):
    torch.manual_seed(0)
    q = torch.randn(batch, nheads, seqlen, headdim, device="cpu", dtype=torch.float32)
    k = torch.randn(batch, nheads, seqlen, headdim, device="cpu", dtype=torch.float32)
    v = torch.randn(batch, nheads, seqlen, headdim, device="cpu", dtype=torch.float32)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    tri = scaled_dot_product_attention(q, k, v, is_causal=True)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("nheads", [2, 4])
@pytest.mark.parametrize("seqlen", [8, 16])
@pytest.mark.parametrize("headdim", [16, 32])
def test_scaled_dot_product_attention_forward(batch, nheads, seqlen, headdim):
    torch.manual_seed(0)
    q = torch.randn(batch, nheads, seqlen, headdim, device="cpu", dtype=torch.float32)
    k = torch.randn(batch, nheads, seqlen, headdim, device="cpu", dtype=torch.float32)
    v = torch.randn(batch, nheads, seqlen, headdim, device="cpu", dtype=torch.float32)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    tri, _ = scaled_dot_product_attention_forward(q, k, v, is_causal=False)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("scale", [None, 0.1, 0.5])
def test_scaled_dot_product_attention_scale(scale):
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 16, device="cpu", dtype=torch.float32)
    k = torch.randn(1, 2, 8, 16, device="cpu", dtype=torch.float32)
    v = torch.randn(1, 2, 8, 16, device="cpu", dtype=torch.float32)

    ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
    tri = scaled_dot_product_attention(q, k, v, scale=scale)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("nheads", [2, 4])
@pytest.mark.parametrize("seqlen", [8, 16, 160])
@pytest.mark.parametrize("headdim", [16, 32])
def test_scaled_dot_product_attention_backward(batch, nheads, seqlen, headdim):
    torch.manual_seed(0)
    q = torch.randn(
        batch,
        nheads,
        seqlen,
        headdim,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    k = torch.randn(
        batch,
        nheads,
        seqlen,
        headdim,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    v = torch.randn(
        batch,
        nheads,
        seqlen,
        headdim,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )

    # Reference backward via torch
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    grad = torch.randn_like(ref)
    ref.backward(grad)
    ref_dq = q.grad.clone()
    ref_dk = k.grad.clone()
    ref_dv = v.grad.clone()

    # Triton backward via ScaleDotProductAttention autograd Function
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)

    tri = ScaleDotProductAttention.apply(q2, k2, v2, None, 0.0, False, None, False)
    tri.backward(grad)

    torch.testing.assert_close(q2.grad, ref_dq, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k2.grad, ref_dk, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v2.grad, ref_dv, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seqlen_q", [8, 16])
@pytest.mark.parametrize("seqlen_k", [8, 16])
@pytest.mark.parametrize("headdim", [32])
def test_flash_attention_forward(batch, seqlen_q, seqlen_k, headdim):
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen_q, 2, headdim, device="cpu", dtype=torch.float16)
    k = torch.randn(batch, seqlen_k, 2, headdim, device="cpu", dtype=torch.float16)
    v = torch.randn(batch, seqlen_k, 2, headdim, device="cpu", dtype=torch.float16)

    # PyTorch reference expects (B, H, L, D)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).to(torch.float32),
        k.transpose(1, 2).to(torch.float32),
        v.transpose(1, 2).to(torch.float32),
        is_causal=False,
    ).to(q.dtype)

    tri_out, tri_q, tri_k, tri_v, lse, seed, offset, p = flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        None,
        None,
        0.0,
        False,
        False,
        scale=None,
    )

    assert tri_out.dtype == q.dtype
    torch.testing.assert_close(tri_out.transpose(1, 2), ref, rtol=1e-3, atol=5e-4)


def test_flash_attention_forward_causal():
    torch.manual_seed(0)
    # Expected layout for flash_attention_forward/mha_fwd: (B, L, H, D)
    q = torch.randn(1, 8, 2, 32, device="cpu", dtype=torch.float16)
    k = torch.randn(1, 8, 2, 32, device="cpu", dtype=torch.float16)
    v = torch.randn(1, 8, 2, 32, device="cpu", dtype=torch.float16)

    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).to(torch.float32),
        k.transpose(1, 2).to(torch.float32),
        v.transpose(1, 2).to(torch.float32),
        is_causal=True,
    ).to(q.dtype)

    tri_out, *_ = flash_attention_forward(
        q, k, v, None, None, None, None, 0.0, True, False, scale=None
    )

    assert tri_out.dtype == q.dtype
    torch.testing.assert_close(tri_out.transpose(1, 2), ref, rtol=1e-3, atol=5e-4)


def test_flash_attention_forward_scale():
    torch.manual_seed(0)
    q = torch.randn(1, 8, 2, 32, device="cpu", dtype=torch.float16)
    k = torch.randn(1, 8, 2, 32, device="cpu", dtype=torch.float16)
    v = torch.randn(1, 8, 2, 32, device="cpu", dtype=torch.float16)

    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).to(torch.float32),
        k.transpose(1, 2).to(torch.float32),
        v.transpose(1, 2).to(torch.float32),
        scale=0.125,
    ).to(q.dtype)

    tri_out, *_ = flash_attention_forward(
        q, k, v, None, None, None, None, 0.0, False, False, scale=0.125
    )

    assert tri_out.dtype == q.dtype
    torch.testing.assert_close(tri_out.transpose(1, 2), ref, rtol=1e-3, atol=5e-4)


def test_flash_attn_varlen_not_supported():
    with pytest.raises(NotImplementedError):
        flash_attn_varlen_func(None, None, None, None, None, 1, None, None, 1.0, False)


def test_flash_attn_varlen_opt_not_supported():
    with pytest.raises(NotImplementedError):
        flash_attn_varlen_opt_func(
            None, None, None, None, None, 1, None, None, 1.0, False
        )
