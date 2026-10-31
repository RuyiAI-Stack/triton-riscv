import pytest
import torch
import triton
import triton.language as tl

from .flash_kernel import apply_dropout

from .flash_attention_backward import (
    _parse_philox,
    efficient_attention_backward,
    flash_attention_backward,
    flash_attn_backward,
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


def _attention_forward_bhsd(
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

    out, _ = _attention_forward_bhsd(
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
    out_bshd, lse = _attention_forward_bhsd(q_bshd, k_bshd, v_bshd)
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


@pytest.mark.parametrize("shape", [(2, 4, 2, 8), (1, 2, 3, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scaled_dot_product_flash_attention_backward_matches_torch(shape, dtype):
    torch.manual_seed(0)
    # This ATen entry point accepts BHSD, with LSE shaped BHS.
    q, k, v = [torch.randn(shape, dtype=dtype) for _ in range(3)]
    grad = torch.randn_like(q)
    out, lse = _attention_forward_bhsd(q, k, v)
    actual = scaled_dot_product_flash_attention_backward(
        grad,
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        shape[2],
        shape[2],
        0.0,
        False,
        torch.tensor(7),
        torch.tensor(11),
    )
    expected = _attention_reference_backward(grad, q, k, v)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)


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
    out_bshd, lse = _attention_forward_bhsd(
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
    out_bshd, lse = _attention_forward_bhsd(
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
    out_bshd, lse = _attention_forward_bhsd(q_bqhd, k_bqhd, v_bqhd, attn_bias=bias)
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


@triton.jit
def _dropout_mask_kernel(
    output,
    length: tl.constexpr,
    heads: tl.constexpr,
    batch: tl.constexpr,
    BLOCK: tl.constexpr,
):
    head = tl.program_id(0)
    keep = apply_dropout(
        tl.full((BLOCK, BLOCK), 1, tl.float32),
        tl.full((), 0, tl.int32),
        tl.full((), 0, tl.int32),
        length,
        batch,
        head,
        tl.full((), 7, tl.uint64),
        tl.full((), 11, tl.uint64),
        127,
        True,
        encode_dropout_in_sign_bit=False,
        NUM_HEADS=heads,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
    )
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    tl.store(
        output + (head * length + rows) * length + cols,
        keep,
        (rows < length) & (cols < length),
    )


def _dropout_keep_mask(length, heads, batch):
    # A synthetic biased attention forward needs the same mask convention;
    # the separate real-forward integration test checks the RNG contract.
    result = torch.empty((1, heads, length, length), dtype=torch.bool)
    _dropout_mask_kernel[(heads,)](
        result, length, heads, batch, max(4, triton.next_power_of_2(length))
    )
    return result


@pytest.mark.parametrize("varlen", [False, True])
def test_flash_attention_backward_dropout_matches_autograd(varlen):
    torch.manual_seed(21)
    lengths = [3, 5] if varlen else [5, 5]
    heads = 2
    max_len = max(lengths)
    qs, ks, vs, outputs, lses, grads, biases, references = ([] for _ in range(8))
    for batch, length in enumerate(lengths):
        q, k, v = [torch.randn(1, heads, length, 16).requires_grad_() for _ in range(3)]
        bias = torch.randn(1, heads, length, length).requires_grad_()
        logits = q @ k.transpose(-1, -2) / 4 + bias
        lse = logits.logsumexp(-1)
        probabilities = torch.softmax(logits, dim=-1)
        keep = _dropout_keep_mask(length, heads, batch)
        assert keep.any() and (~keep).any()
        out = (probabilities * keep * 2) @ v
        grad = torch.randn_like(out)
        out.backward(grad)
        qs.append(_to_bshd(q.detach()))
        ks.append(_to_bshd(k.detach()))
        vs.append(_to_bshd(v.detach()))
        outputs.append(_to_bshd(out.detach()))
        lses.append(lse.detach())
        grads.append(_to_bshd(grad))
        biases.append(bias.detach())
        references.append((q.grad, k.grad, v.grad, bias.grad))

    def pack(tensors):
        if varlen:
            return torch.cat([x.squeeze(0) for x in tensors])
        return torch.cat(tensors)

    if varlen:
        lse = torch.cat([x.squeeze(0).transpose(0, 1) for x in lses])
        bias = torch.cat(
            [
                torch.nn.functional.pad(
                    x.squeeze(0).transpose(0, 1), (0, max_len - length)
                )
                for x, length in zip(biases, lengths)
            ]
        )
        cumulative = torch.tensor([0, 3, 8], dtype=torch.int32)
    else:
        lse = torch.cat(lses)
        bias = torch.cat(biases)
        cumulative = None
    actual = flash_attn_backward(
        pack(grads),
        pack(qs),
        pack(ks),
        pack(vs),
        pack(outputs),
        lse,
        cu_seq_q=cumulative,
        cu_seq_k=cumulative,
        max_seqlen_q=max_len,
        max_seqlen_k=max_len,
        is_dropout=True,
        dropout_p=0.5,
        rng_state=(7, 11),
        attn_bias=bias,
        bias_requires_grad=True,
    )
    for component in range(3):
        expected = pack([_to_bshd(ref[component]) for ref in references])
        torch.testing.assert_close(actual[component], expected, rtol=1e-4, atol=1e-4)
    if varlen:
        expected_bias = torch.cat(
            [
                torch.nn.functional.pad(
                    ref[3].squeeze(0).transpose(0, 1), (0, max_len - length)
                )
                for ref, length in zip(references, lengths)
            ]
        )
    else:
        expected_bias = torch.cat([ref[3] for ref in references])
    torch.testing.assert_close(actual[3], expected_bias, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mask_kind", ["causal", "window"])
def test_flash_attention_bias_gradient_skipped_tiles(monkeypatch, mask_kind):
    torch.manual_seed(22)
    causal = mask_kind == "causal"
    # Three tiles ensure a nonzero-width window skips a whole distant tile.
    length = 128 if causal else 192
    q, k, v = [torch.randn(1, 1, length, 16).requires_grad_() for _ in range(3)]
    bias = torch.randn(1, 1, length, length).requires_grad_()
    window = -1 if causal else 3
    out, lse = _attention_forward_bhsd(
        q,
        k,
        v,
        attn_bias=bias,
        is_causal=causal,
        window_size_left=window,
        window_size_right=window,
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    original_empty_like = torch.empty_like

    def poisoned_empty_like(tensor, *args, **kwargs):
        result = original_empty_like(tensor, *args, **kwargs)
        if tensor.shape == bias.shape:
            result.fill_(12345)
        return result

    monkeypatch.setattr(torch, "empty_like", poisoned_empty_like)
    actual = flash_attn_backward(
        _to_bshd(grad),
        _to_bshd(q.detach()),
        _to_bshd(k.detach()),
        _to_bshd(v.detach()),
        _to_bshd(out.detach()),
        lse.detach(),
        is_causal=causal,
        window_size_left=window,
        window_size_right=window,
        attn_bias=bias.detach(),
        bias_requires_grad=True,
    )
    torch.testing.assert_close(actual[3], bias.grad, rtol=1e-4, atol=1e-4)
    rows = torch.arange(length)[:, None]
    cols = torch.arange(length)[None, :]
    excluded = (cols > rows) if causal else ((rows - cols).abs() > window)
    assert torch.count_nonzero(actual[3][0, 0][excluded]) == 0


@pytest.mark.parametrize(
    "varlen,heads,kv_heads,causal,long_k",
    [
        (varlen, heads, kv_heads, causal, False)
        for causal in (False, True)
        for heads, kv_heads in ((2, 2), (4, 2))
        for varlen in (False, True)
    ]
    + [pytest.param(False, 2, 2, False, True, id="second-key-tile")],
)
def test_flash_attention_dropout_forward_backward(
    monkeypatch, varlen, heads, kv_heads, causal, long_k
):
    from . import flash_api

    torch.manual_seed(29)
    # Exercise nonzero batch/head counters, GQA, tails and multiple forward tiles.
    lengths = [17, 33] if varlen else [33, 33]
    key_lengths = lengths if causal else ([9, 33] if varlen else [33, 33])
    if not causal:
        lengths = [17, 65] if varlen else [65, 65]
    if long_k:
        # The second backward key tile must use col_start=64 in the RNG stream.
        key_lengths = [65, 65]
    dim = 128 if long_k else 64
    dropout_p = 0.25
    seed, offset = (1 << 40) + 7, (1 << 35) + 11
    monkeypatch.setattr(
        flash_api, "philox_backend_seed_offset", lambda increment: (seed, offset)
    )
    qs, ks, vs = [], [], []
    for length, key_length in zip(lengths, key_lengths):
        qs.append(torch.randn(length, heads, dim, dtype=torch.float16) * 0.25)
        ks.append(torch.randn(key_length, kv_heads, dim, dtype=torch.float16) * 0.25)
        # Identity values expose the actual forward dropout mask in the output:
        # output[row, col] is the retained attention probability for that column.
        vs.append(
            torch.eye(key_length, dim, dtype=torch.float16)[:, None, :].repeat(
                1, kv_heads, 1
            )
        )
    pack = torch.cat if varlen else torch.stack
    q, k, v = pack(qs), pack(ks), pack(vs)
    cumulative = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
    cumulative_k = torch.tensor(
        [0, key_lengths[0], sum(key_lengths)], dtype=torch.int32
    )
    if varlen:
        result = flash_api.mha_varlan_fwd(
            q,
            k,
            v,
            None,
            cumulative,
            cumulative_k,
            None,
            None,
            None,
            None,
            max(lengths),
            max(key_lengths),
            dropout_p,
            dim**-0.5,
            False,
            causal,
            -1,
            -1,
            0.0,
            False,
            None,
        )
        out, lse, rng_state = result[0], result[4], result[5]
        backward_lse = lse.transpose(0, 1).contiguous()
        outputs = out.split(lengths)
    else:
        result = flash_api.mha_fwd(
            q,
            k,
            v,
            None,
            None,
            dropout_p,
            dim**-0.5,
            causal,
            -1,
            -1,
            0.0,
            False,
        )
        out, backward_lse, rng_state = result[0], result[4], result[5]
        outputs = out.unbind()
    assert tuple(rng_state.tolist()) == (seed, offset)
    references, grads = [], []
    for batch, (q_part, k_part, v_part, output) in enumerate(zip(qs, ks, vs, outputs)):
        q_ref, k_ref, v_ref = [
            x.float().detach().requires_grad_() for x in (q_part, k_part, v_part)
        ]
        k_expanded = k_ref.repeat_interleave(heads // kv_heads, dim=1)
        v_expanded = v_ref.repeat_interleave(heads // kv_heads, dim=1)
        logits = torch.einsum("mhd,nhd->hmn", q_ref, k_expanded) * dim**-0.5
        if causal:
            excluded = torch.ones(
                lengths[batch], key_lengths[batch], dtype=torch.bool
            ).triu(1)
            logits = logits.masked_fill(excluded, float("-inf"))
        probabilities = logits.softmax(-1)
        keep = output[:, :, : key_lengths[batch]].permute(1, 0, 2) > 0
        assert keep.any() and (~keep).any()
        reference = torch.einsum(
            "hmn,nhd->mhd", probabilities * keep / (1 - dropout_p), v_expanded
        )
        torch.testing.assert_close(output.float(), reference, atol=2e-3, rtol=2e-3)
        expected_lse = logits.logsumexp(-1)
        actual_lse = (
            lse[:, cumulative[batch] : cumulative[batch + 1]]
            if varlen
            else backward_lse[batch]
        )
        torch.testing.assert_close(actual_lse, expected_lse, atol=2e-3, rtol=2e-3)
        grad = torch.randn_like(output)
        reference.backward(grad.float())
        grads.append(grad)
        references.append((q_ref.grad, k_ref.grad, v_ref.grad))
    actual = flash_attn_backward(
        pack(grads),
        q,
        k,
        v,
        out,
        backward_lse,
        cu_seq_q=cumulative if varlen else None,
        cu_seq_k=cumulative_k if varlen else None,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(key_lengths),
        is_causal=causal,
        is_dropout=True,
        dropout_p=dropout_p,
        rng_state=tuple(rng_state.tolist()),
    )
    for component in range(3):
        expected = pack([ref[component] for ref in references])
        torch.testing.assert_close(
            actual[component].float(), expected, atol=3e-3, rtol=1e-2
        )
