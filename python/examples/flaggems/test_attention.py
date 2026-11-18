import pytest
import torch
import torch.nn.functional as F

from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    flash_attn_varlen_opt_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)


def _flash_attention_forward_output(*args, **kwargs):
    result = flash_attention_forward(*args, **kwargs)
    assert isinstance(result, tuple)
    assert len(result) == 5
    output, _, _, _, _ = result
    return output


def _make_varlen_inputs():
    torch.manual_seed(0)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device="cpu")
    q = torch.randn(8, 2, 32, device="cpu", dtype=torch.float16)
    k = torch.randn(8, 2, 32, device="cpu", dtype=torch.float16)
    v = torch.randn(8, 2, 32, device="cpu", dtype=torch.float16)
    return q, k, v, cu_seqlens


def _varlen_reference(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k=None,
    causal=False,
    scale=None,
    window_size=None,
):
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    outs = []
    q_ranges = zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist())
    k_ranges = zip(cu_seqlens_k[:-1].tolist(), cu_seqlens_k[1:].tolist())
    for (q_start, q_end), (k_start, k_end) in zip(q_ranges, k_ranges):
        q_batch = q[q_start:q_end].transpose(0, 1).unsqueeze(0).to(torch.float32)
        k_batch = k[k_start:k_end].transpose(0, 1).unsqueeze(0).to(torch.float32)
        v_batch = v[k_start:k_end].transpose(0, 1).unsqueeze(0).to(torch.float32)
        attn_mask = None
        if window_size is not None:
            window_left, window_right = window_size
            q_len = q_end - q_start
            k_len = k_end - k_start
            q_idx = torch.arange(q_len)[:, None]
            k_idx = torch.arange(k_len)[None, :]
            diagonal = k_len - q_len
            attn_mask = torch.ones((q_len, k_len), dtype=torch.bool)
            if window_left >= 0:
                attn_mask &= k_idx >= q_idx + diagonal - window_left
            if window_right >= 0:
                attn_mask &= k_idx <= q_idx + diagonal + window_right
        out = F.scaled_dot_product_attention(
            q_batch,
            k_batch,
            v_batch,
            attn_mask=attn_mask,
            is_causal=causal,
            scale=scale,
        )
        outs.append(out.squeeze(0).transpose(0, 1).to(q.dtype))
    return torch.cat(outs, dim=0)


def _assert_flash_close(actual, reference):
    reference = reference.to(actual.dtype)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)


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
@pytest.mark.parametrize("seqlen", [8, 16])
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

    # Directly validate the exported backward entrypoint.
    q2 = q.detach().clone()
    k2 = k.detach().clone()
    v2 = v.detach().clone()
    tri, softmax_lse = scaled_dot_product_attention_forward(q2, k2, v2, is_causal=False)
    tri_dq, tri_dk, tri_dv = scaled_dot_product_attention_backward(
        grad.contiguous(),
        q2,
        k2,
        v2,
        tri.contiguous(),
        softmax_lse,
        is_causal=False,
    )

    torch.testing.assert_close(tri_dq, ref_dq, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_dk, ref_dk, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_dv, ref_dv, rtol=1e-3, atol=1e-3)


def test_scaled_dot_product_attention_backward_cross_attention_longer_kv():
    torch.manual_seed(0)
    batch, nheads, seqlen_q, seqlen_kv, headdim = 2, 2, 128, 256, 16
    q = torch.randn(
        batch,
        nheads,
        seqlen_q,
        headdim,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    k = torch.randn(
        batch,
        nheads,
        seqlen_kv,
        headdim,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    v = torch.randn(
        batch,
        nheads,
        seqlen_kv,
        headdim,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    grad = torch.randn_like(ref)
    ref.backward(grad)
    ref_dq = q.grad.clone()
    ref_dk = k.grad.clone()
    ref_dv = v.grad.clone()

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
    )

    tri_out = _flash_attention_forward_output(
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

    _assert_flash_close(tri_out.transpose(1, 2), ref)


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
    )

    tri_out = _flash_attention_forward_output(
        q, k, v, None, None, None, None, 0.0, True, False, scale=None
    )

    _assert_flash_close(tri_out.transpose(1, 2), ref)


@pytest.mark.parametrize("scale", [0.125, 0.0])
def test_flash_attention_forward_scale(scale):
    torch.manual_seed(0)
    q = torch.randn(1, 8, 2, 32, device="cpu", dtype=torch.float16)
    k = torch.randn(1, 8, 2, 32, device="cpu", dtype=torch.float16)
    v = torch.randn(1, 8, 2, 32, device="cpu", dtype=torch.float16)

    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).to(torch.float32),
        k.transpose(1, 2).to(torch.float32),
        v.transpose(1, 2).to(torch.float32),
        scale=scale,
    )

    tri_out = _flash_attention_forward_output(
        q, k, v, None, None, None, None, 0.0, False, False, scale=scale
    )

    _assert_flash_close(tri_out.transpose(1, 2), ref)


def test_flash_attn_varlen_func():
    q, k, v, cu_seqlens = _make_varlen_inputs()

    ref_out = _varlen_reference(q, k, v, cu_seqlens)
    for _ in range(5):
        tri_out = flash_attn_varlen_func(
            q,
            k,
            v,
            5,
            cu_seqlens,
            5,
            cu_seqlens_k=cu_seqlens,
            dropout_p=0.0,
            causal=False,
        )
        _assert_flash_close(tri_out, ref_out)


def test_flash_attn_varlen_func_empty_queries_return_empty_lse():
    cu_seqlens_q = torch.tensor([0, 0, 0], dtype=torch.int32, device="cpu")
    cu_seqlens_k = torch.tensor([0, 2, 5], dtype=torch.int32, device="cpu")
    q = torch.empty(0, 2, 32, device="cpu", dtype=torch.float16)
    k = torch.randn(5, 2, 32, device="cpu", dtype=torch.float16)
    v = torch.randn(5, 2, 32, device="cpu", dtype=torch.float16)

    tri_out, softmax_lse = flash_attn_varlen_func(
        q,
        k,
        v,
        0,
        cu_seqlens_q,
        3,
        cu_seqlens_k=cu_seqlens_k,
        dropout_p=0.0,
        causal=False,
        return_softmax_lse=True,
    )

    assert tri_out.shape == q.shape
    assert isinstance(softmax_lse, torch.Tensor)
    assert softmax_lse.shape == (q.shape[1], q.shape[0])
    assert softmax_lse.dtype == torch.float32
    assert softmax_lse.device == q.device


def test_flash_attn_varlen_opt_func():
    q, k, v, cu_seqlens = _make_varlen_inputs()

    ref_out = _varlen_reference(q, k, v, cu_seqlens)
    for _ in range(5):
        tri_out = flash_attn_varlen_opt_func(
            q,
            k,
            v,
            5,
            cu_seqlens,
            5,
            cu_seqlens_k=cu_seqlens,
            dropout_p=0.0,
            causal=False,
        )
        _assert_flash_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "implementation", [flash_attn_varlen_func, flash_attn_varlen_opt_func]
)
def test_flash_attn_varlen_uses_k_sequence_offsets(implementation):
    torch.manual_seed(0)
    cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 3, 7], dtype=torch.int32)
    q = torch.randn(5, 2, 32, dtype=torch.float16)
    k = torch.randn(7, 2, 32, dtype=torch.float16)
    v = torch.randn(7, 2, 32, dtype=torch.float16)
    ref_out = _varlen_reference(q, k, v, cu_seqlens_q, cu_seqlens_k)

    tri_out = implementation(
        q,
        k,
        v,
        3,
        cu_seqlens_q,
        4,
        cu_seqlens_k=cu_seqlens_k,
    )

    _assert_flash_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "implementation", [flash_attn_varlen_func, flash_attn_varlen_opt_func]
)
def test_flash_attn_varlen_causal_window(implementation):
    q, k, v, cu_seqlens = _make_varlen_inputs()
    ref_out = _varlen_reference(q, k, v, cu_seqlens, causal=True)

    tri_out = implementation(
        q,
        k,
        v,
        5,
        cu_seqlens,
        5,
        cu_seqlens_k=cu_seqlens,
        window_size=(-1, 0),
    )

    _assert_flash_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "implementation", [flash_attn_varlen_func, flash_attn_varlen_opt_func]
)
def test_flash_attn_varlen_one_sided_window(implementation):
    q, k, v, cu_seqlens = _make_varlen_inputs()
    ref_out = _varlen_reference(q, k, v, cu_seqlens, window_size=(2, -1))

    tri_out = implementation(
        q,
        k,
        v,
        5,
        cu_seqlens,
        5,
        cu_seqlens_k=cu_seqlens,
        window_size=(2, -1),
    )

    _assert_flash_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "implementation", [flash_attn_varlen_func, flash_attn_varlen_opt_func]
)
def test_flash_attn_varlen_noncontiguous_last_dim(implementation):
    torch.manual_seed(0)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    q = torch.randn(8, 2, 64, dtype=torch.float16)[..., ::2]
    k = torch.randn(8, 2, 64, dtype=torch.float16)[..., ::2]
    v = torch.randn(8, 2, 64, dtype=torch.float16)[..., ::2]
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 2
    ref_out = _varlen_reference(q, k, v, cu_seqlens)

    tri_out = implementation(q, k, v, 5, cu_seqlens, 5, cu_seqlens_k=cu_seqlens)

    _assert_flash_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "implementation", [flash_attn_varlen_func, flash_attn_varlen_opt_func]
)
def test_flash_attn_varlen_rejects_seqused_without_paging(implementation):
    q, k, v, cu_seqlens = _make_varlen_inputs()
    with pytest.raises(AssertionError, match="requires cu_seqlens_k"):
        implementation(
            q,
            k,
            v,
            5,
            cu_seqlens,
            5,
            seqused_k=torch.tensor([3, 5], dtype=torch.int32),
        )


@pytest.mark.parametrize(
    "implementation", [flash_attn_varlen_func, flash_attn_varlen_opt_func]
)
def test_flash_attn_varlen_rejects_unimplemented_attention_probs(implementation):
    q, k, v, cu_seqlens = _make_varlen_inputs()
    with pytest.raises(NotImplementedError, match="return_attn_probs"):
        implementation(
            q,
            k,
            v,
            5,
            cu_seqlens,
            5,
            cu_seqlens_k=cu_seqlens,
            return_attn_probs=True,
        )
