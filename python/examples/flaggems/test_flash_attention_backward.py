import pytest
import torch

from .flash_attention_backward import (
    _parse_philox,
    efficient_attention_backward,
    flash_attention_backward,
    scaled_dot_product_cudnn_attention_backward,
    scaled_dot_product_efficient_attention_backward,
    scaled_dot_product_flash_attention_backward,
)


def _assert_attention_match(custom_fn, torch_fn, *args, **kwargs):
    tri_out = custom_fn(*args, **kwargs)
    ref_out = torch_fn(*args, **kwargs)

    if isinstance(tri_out, tuple):
        assert len(tri_out) == len(ref_out)
        for tri, ref in zip(tri_out, ref_out):
            if tri is None or ref is None:
                assert tri is None
            else:
                torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)


def _attention_forward_bshd(
    query,
    key,
    value,
    *,
    attn_bias=None,
    scale=None,
    is_causal=False,
    window_size_left=None,
    window_size_right=None,
):
    scale = (1.0 / query.shape[-1] ** 0.5) if scale is None else scale
    logits = torch.matmul(query, key.transpose(-1, -2)) * scale
    if attn_bias is not None:
        logits = logits + attn_bias

    seq_q = logits.shape[-2]
    seq_k = logits.shape[-1]
    rows = torch.arange(seq_q, device=query.device).view(1, 1, seq_q, 1)
    cols = torch.arange(seq_k, device=query.device).view(1, 1, 1, seq_k)
    dist = rows - cols
    mask = torch.ones_like(logits, dtype=torch.bool)
    if is_causal:
        mask = mask & (dist >= 0)
    if window_size_left is not None and window_size_left >= 0:
        mask = mask & (dist <= int(window_size_left))
    if window_size_right is not None and window_size_right >= 0:
        mask = mask & (dist >= -int(window_size_right))

    logits = logits.masked_fill(~mask, float("-inf"))
    logsumexp = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(logits - logsumexp.unsqueeze(-1))
    out = torch.matmul(probs.to(value.dtype), value)
    return out, logsumexp


def _attention_reference_backward(
    grad_out,
    query,
    key,
    value,
    *,
    attn_bias=None,
    scale=None,
    is_causal=False,
    window_size_left=None,
    window_size_right=None,
):
    q = query.detach().clone().to(torch.float32).requires_grad_(True)
    k = key.detach().clone().to(torch.float32).requires_grad_(True)
    v = value.detach().clone().to(torch.float32).requires_grad_(True)
    grad = grad_out.to(torch.float32)

    out, _ = _attention_forward_bshd(
        q,
        k,
        v,
        attn_bias=attn_bias,
        scale=scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    out.backward(grad)
    return (
        q.grad.to(query.dtype),
        k.grad.to(key.dtype),
        v.grad.to(value.dtype),
    )


def _to_bshd(x):
    return x.permute(0, 2, 1, 3).contiguous()


def _to_bqsd(x):
    return x.permute(0, 2, 1, 3).contiguous()


def test_parse_philox():
    torch.manual_seed(0)
    seed, offset = _parse_philox(
        torch.tensor(7, device="cpu"), torch.tensor(11, device="cpu")
    )

    assert seed == 7
    assert offset == 11


@pytest.mark.parametrize("shape", [(2, 4, 2, 8), (1, 2, 4, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_flash_attention_backward_matches_torch(shape, dtype):
    torch.manual_seed(0)
    device = "cpu"
    q = torch.randn(*shape, device=device, dtype=dtype)
    k = torch.randn(*shape, device=device, dtype=dtype)
    v = torch.randn(*shape, device=device, dtype=dtype)
    grad = torch.randn_like(q)

    q_bshd = _to_bshd(q)
    k_bshd = _to_bshd(k)
    v_bshd = _to_bshd(v)
    out_bshd, lse = _attention_forward_bshd(q_bshd, k_bshd, v_bshd)
    out = _to_bqsd(out_bshd)

    def torch_ref(
        grad_out,
        query,
        key,
        value,
        attn_out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        rng_state,
        unused,
        *,
        scale=None,
        window_size_left=None,
        window_size_right=None,
    ):
        del attn_out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k
        del dropout_p, is_causal, rng_state, unused
        del scale, window_size_left, window_size_right

        grad_bshd = _to_bshd(grad_out)
        q_bshd = _to_bshd(query)
        k_bshd = _to_bshd(key)
        v_bshd = _to_bshd(value)
        d_q, d_k, d_v = _attention_reference_backward(grad_bshd, q_bshd, k_bshd, v_bshd)
        return (
            _to_bqsd(d_q),
            _to_bqsd(d_k),
            _to_bqsd(d_v),
        )

    _assert_attention_match(
        flash_attention_backward,
        torch_ref,
        grad,
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        shape[1],
        shape[1],
        0.0,
        False,
        torch.tensor([7], device=device, dtype=torch.int64),
        torch.tensor(0, device=device, dtype=torch.int64),
    )


@pytest.mark.parametrize("shape", [(2, 4, 2, 8), (1, 2, 4, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scaled_dot_product_flash_attention_backward_matches_torch(shape, dtype):
    torch.manual_seed(0)
    device = "cpu"
    q = torch.randn(*shape, device=device, dtype=dtype)
    k = torch.randn(*shape, device=device, dtype=dtype)
    v = torch.randn(*shape, device=device, dtype=dtype)
    grad = torch.randn_like(q)

    q_bshd = _to_bshd(q)
    k_bshd = _to_bshd(k)
    v_bshd = _to_bshd(v)
    out_bshd, lse = _attention_forward_bshd(q_bshd, k_bshd, v_bshd)
    out = _to_bqsd(out_bshd)

    def torch_ref(
        grad_out,
        query,
        key,
        value,
        attn_out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        *,
        scale=None,
    ):
        del attn_out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k
        del dropout_p, is_causal, philox_seed, philox_offset, scale
        grad_bshd = _to_bshd(grad_out)
        q_bshd = _to_bshd(query)
        k_bshd = _to_bshd(key)
        v_bshd = _to_bshd(value)
        d_q, d_k, d_v = _attention_reference_backward(grad_bshd, q_bshd, k_bshd, v_bshd)
        return (
            _to_bqsd(d_q),
            _to_bqsd(d_k),
            _to_bqsd(d_v),
        )

    _assert_attention_match(
        scaled_dot_product_flash_attention_backward,
        torch_ref,
        grad,
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        shape[1],
        shape[1],
        0.0,
        False,
        torch.tensor([7], device=device, dtype=torch.int64),
        torch.tensor([11], device=device, dtype=torch.int64),
    )


@pytest.mark.parametrize("shape", [(2, 2, 4, 8), (1, 1, 8, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scaled_dot_product_efficient_attention_backward_matches_torch(shape, dtype):
    torch.manual_seed(0)
    device = "cpu"
    q = torch.randn(*shape, device=device, dtype=dtype)
    k = torch.randn(*shape, device=device, dtype=dtype)
    v = torch.randn(*shape, device=device, dtype=dtype)
    grad = torch.randn_like(q)
    bias = torch.zeros(
        shape[0], shape[1], shape[2], shape[2], device=device, dtype=torch.float32
    )

    q_bqhd = q
    k_bqhd = k
    v_bqhd = v
    out_bshd, lse = _attention_forward_bshd(
        q_bqhd,
        k_bqhd,
        v_bqhd,
        attn_bias=bias,
    )
    out = out_bshd

    def torch_ref(
        grad_out_,
        query,
        key,
        value,
        attn_bias,
        out_tensor,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        grad_input_mask,
        is_causal=False,
        *,
        scale=None,
    ):
        del (
            out_tensor,
            logsumexp,
            philox_seed,
            philox_offset,
            dropout_p,
            is_causal,
            scale,
        )
        grad_bqhd = grad_out_
        q_bqhd = query
        k_bqhd = key
        v_bqhd = value
        d_q, d_k, d_v = _attention_reference_backward(
            grad_bqhd,
            q_bqhd,
            k_bqhd,
            v_bqhd,
            attn_bias=attn_bias,
        )
        if grad_input_mask[3]:
            dbias = torch.zeros_like(attn_bias)
        else:
            dbias = None
        return d_q, d_k, d_v, dbias

    _assert_attention_match(
        scaled_dot_product_efficient_attention_backward,
        torch_ref,
        grad,
        q,
        k,
        v,
        bias,
        out,
        lse,
        torch.tensor([7], device=device, dtype=torch.int64),
        torch.tensor([11], device=device, dtype=torch.int64),
        0.0,
        [True, True, True, False],
        False,
    )


@pytest.mark.parametrize("shape", [(2, 2, 2, 8), (1, 1, 4, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_efficient_attention_backward_matches_torch(shape, dtype):
    torch.manual_seed(0)
    device = "cpu"
    q = torch.randn(*shape, device=device, dtype=dtype)
    k = torch.randn(*shape, device=device, dtype=dtype)
    v = torch.randn(*shape, device=device, dtype=dtype)
    grad = torch.randn_like(q)
    bias = torch.zeros(
        shape[0], shape[2], shape[1], shape[1], device=device, dtype=torch.float32
    )

    q_bshd = _to_bshd(q)
    k_bshd = _to_bshd(k)
    v_bshd = _to_bshd(v)
    attn_bias_bshd = bias
    out_bshd, lse = _attention_forward_bshd(
        q_bshd,
        k_bshd,
        v_bshd,
        attn_bias=attn_bias_bshd,
    )
    out = _to_bqsd(out_bshd)

    def torch_ref(
        grad_out_,
        query,
        key,
        value,
        attn_bias,
        out_tensor,
        logsumexp,
        cum_seqlens_q,
        cum_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        philox_seed,
        philox_offset,
        custom_mask_type,
        bias_requires_grad,
        *,
        scale=None,
        num_splits_key=None,
        window_size=None,
        shared_storage_dqdkdv=False,
    ):
        del (
            attn_bias,
            out_tensor,
            logsumexp,
            cum_seqlens_q,
            cum_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            philox_seed,
            philox_offset,
            custom_mask_type,
            bias_requires_grad,
            scale,
            num_splits_key,
            window_size,
            shared_storage_dqdkdv,
        )
        grad_bshd = _to_bshd(grad_out_)
        q_bshd = _to_bshd(query)
        k_bshd = _to_bshd(key)
        v_bshd = _to_bshd(value)
        d_q, d_k, d_v = _attention_reference_backward(grad_bshd, q_bshd, k_bshd, v_bshd)
        return _to_bqsd(d_q), _to_bqsd(d_k), _to_bqsd(d_v), None

    _assert_attention_match(
        efficient_attention_backward,
        torch_ref,
        grad,
        q,
        k,
        v,
        bias,
        out,
        None,
        None,
        shape[2],
        shape[2],
        lse,
        0.0,
        torch.tensor([7], device=device, dtype=torch.int64),
        torch.tensor([11], device=device, dtype=torch.int64),
        0,
        False,
        scale=None,
        num_splits_key=None,
        window_size=None,
        shared_storage_dqdkdv=False,
    )


@pytest.mark.parametrize("shape", [(2, 2, 4, 8), (1, 1, 8, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scaled_dot_product_cudnn_attention_backward_matches_torch(shape, dtype):
    torch.manual_seed(0)
    device = "cpu"
    q = torch.randn(*shape, device=device, dtype=dtype)
    k = torch.randn(*shape, device=device, dtype=dtype)
    v = torch.randn(*shape, device=device, dtype=dtype)
    grad = torch.randn_like(q)
    bias = torch.zeros(
        shape[0], shape[1], shape[2], shape[2], device=device, dtype=torch.float32
    )

    q_bqhd = q
    k_bqhd = k
    v_bqhd = v
    out_bshd, lse = _attention_forward_bshd(q_bqhd, k_bqhd, v_bqhd, attn_bias=bias)
    out = out_bshd

    def torch_ref(
        grad_out,
        query,
        key,
        value,
        attn_out,
        logsumexp,
        philox_seed,
        philox_offset,
        attn_bias,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        *,
        scale=None,
        bias_requires_grad=False,
    ):
        del (
            attn_out,
            logsumexp,
            philox_seed,
            philox_offset,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            scale,
            bias_requires_grad,
        )
        grad_bqhd = grad_out
        q_bqhd = query
        k_bqhd = key
        v_bqhd = value
        d_q, d_k, d_v = _attention_reference_backward(
            grad_bqhd,
            q_bqhd,
            k_bqhd,
            v_bqhd,
            attn_bias=attn_bias,
        )
        return d_q, d_k, d_v

    _assert_attention_match(
        scaled_dot_product_cudnn_attention_backward,
        torch_ref,
        grad,
        q,
        k,
        v,
        out,
        lse,
        torch.tensor([7], device=device, dtype=torch.int64),
        torch.tensor([11], device=device, dtype=torch.int64),
        bias,
        None,
        None,
        shape[2],
        shape[2],
        0.0,
        False,
        scale=None,
    )
