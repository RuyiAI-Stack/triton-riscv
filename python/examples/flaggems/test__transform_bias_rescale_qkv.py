import importlib

import pytest
import torch

from ._transform_bias_rescale_qkv import _transform_bias_rescale_qkv

# torch reference: aten::_transform_bias_rescale_qkv, CPU-callable on this
# build (the flash-attention preprocessing op).
torch_tbq = torch._transform_bias_rescale_qkv

# (batch, seq_len, num_heads, head_dim) covering the regular matrix and the
# degenerate corners (single batch / single step / single head / head_dim 1).
SHAPES = [
    (2, 5, 4, 2),
    (1, 1, 1, 1),
    (1, 7, 1, 8),
    (3, 1, 2, 4),
    (2, 5, 1, 1),
    (1, 33, 8, 4),  # odd seq_len, multi-block (>1024 elements)
    (4, 16, 16, 64),  # large: 65536 elements per section
]


def make_inputs(batch, seq_len, num_heads, head_dim, dtype=torch.float32):
    torch.manual_seed(0)
    qkv = torch.randn(batch, seq_len, 3 * num_heads * head_dim, dtype=dtype)
    bias = torch.randn(3 * num_heads * head_dim, dtype=dtype)
    return qkv, bias


@pytest.mark.parametrize("batch, seq_len, num_heads, head_dim", SHAPES)
def test_transform_bias_rescale_qkv_matches_aten(batch, seq_len, num_heads, head_dim):
    qkv, bias = make_inputs(batch, seq_len, num_heads, head_dim)

    rq, rk, rv = torch_tbq(qkv, bias, num_heads)
    q, k, v = _transform_bias_rescale_qkv(qkv, bias, num_heads)

    assert q.shape == (batch, num_heads, seq_len, head_dim)
    assert k.shape == (batch, num_heads, seq_len, head_dim)
    assert v.shape == (batch, num_heads, seq_len, head_dim)
    for res, ref in ((q, rq), (k, rk), (v, rv)):
        assert res.dtype == ref.dtype
        torch.testing.assert_close(res, ref, rtol=0, atol=0)


@pytest.mark.parametrize("batch, seq_len, num_heads, head_dim", SHAPES[:3])
def test_transform_bias_rescale_qkv_query_scaling_semantics(
    batch, seq_len, num_heads, head_dim
):
    # Only the query section is rescaled, by 1/sqrt(head_dim); k and v only
    # take their bias. Sections pack [Q | K | V] along the last dim.
    qkv, bias = make_inputs(batch, seq_len, num_heads, head_dim)
    scale = head_dim**-0.5

    q, k, v = _transform_bias_rescale_qkv(qkv, bias, num_heads)

    qkv5 = qkv.view(batch, seq_len, 3, num_heads, head_dim)
    b5 = bias.view(3, num_heads, head_dim)
    q_ref = ((qkv5[..., 0, :, :] + b5[0]) * scale).permute(0, 2, 1, 3)
    k_ref = (qkv5[..., 1, :, :] + b5[1]).permute(0, 2, 1, 3)
    v_ref = (qkv5[..., 2, :, :] + b5[2]).permute(0, 2, 1, 3)
    torch.testing.assert_close(q, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(k, k_ref, rtol=0, atol=0)
    torch.testing.assert_close(v, v_ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_transform_bias_rescale_qkv_dtype(dtype):
    batch, seq_len, num_heads, head_dim = 2, 5, 4, 2
    qkv, bias = make_inputs(batch, seq_len, num_heads, head_dim, dtype)

    rq, rk, rv = torch_tbq(qkv, bias, num_heads)
    q, k, v = _transform_bias_rescale_qkv(qkv, bias, num_heads)

    assert q.dtype == dtype
    for res, ref in ((q, rq), (k, rk), (v, rv)):
        torch.testing.assert_close(res, ref, rtol=1e-3, atol=1e-3)


def test_transform_bias_rescale_qkv_non_contiguous_inputs():
    # A transposed qkv view and a strided bias are both routed through the
    # kernel's strides / the contiguous op.
    torch.manual_seed(1)
    qkv = torch.randn(5, 2, 24).transpose(0, 1)
    bias = torch.randn(48)[::2]
    assert not qkv.is_contiguous()
    assert not bias.is_contiguous()

    rq, rk, rv = torch_tbq(qkv, bias, 4)
    q, k, v = _transform_bias_rescale_qkv(qkv, bias, 4)
    for res, ref in ((q, rq), (k, rk), (v, rv)):
        torch.testing.assert_close(res, ref, rtol=0, atol=0)


def test_transform_bias_rescale_qkv_zero_bias():
    torch.manual_seed(0)
    qkv = torch.randn(2, 5, 24)
    bias = torch.zeros(24)

    rq, rk, rv = torch_tbq(qkv, bias, 4)
    q, k, v = _transform_bias_rescale_qkv(qkv, bias, 4)
    for res, ref in ((q, rq), (k, rk), (v, rv)):
        torch.testing.assert_close(res, ref, rtol=0, atol=0)


def test_transform_bias_rescale_qkv_empty_seq():
    # seq_len = 0: zero-element grid launch, empty outputs. aten agrees.
    qkv = torch.randn(2, 0, 24)
    bias = torch.randn(24)

    rq, rk, rv = torch_tbq(qkv, bias, 4)
    q, k, v = _transform_bias_rescale_qkv(qkv, bias, 4)
    assert q.shape == (2, 4, 0, 2)
    for res, ref in ((q, rq), (k, rk), (v, rv)):
        assert res.shape == ref.shape
        assert res.numel() == 0


def test_transform_bias_rescale_qkv_empty_batch():
    # batch = 0: the port returns empty outputs; the aten reference
    # segfaults natively on this build (probed), so no comparison.
    qkv = torch.randn(0, 5, 24)
    bias = torch.randn(24)

    q, k, v = _transform_bias_rescale_qkv(qkv, bias, 4)
    assert q.shape == (0, 4, 5, 2)
    assert k.shape == (0, 4, 5, 2)
    assert v.shape == (0, 4, 5, 2)


def test_transform_bias_rescale_qkv_rejects_non_dividing_heads():
    # aten: "Expected D % num_head == 0"; the port mirrors the guard
    # (the upstream module would silently truncate).
    qkv = torch.randn(2, 5, 24)
    bias = torch.randn(24)
    with pytest.raises(RuntimeError):
        torch_tbq(qkv, bias, 5)
    with pytest.raises(RuntimeError):
        _transform_bias_rescale_qkv(qkv, bias, 5)


def test_transform_bias_rescale_qkv_input_not_modified():
    qkv, bias = make_inputs(2, 5, 4, 2)
    qkv_before = qkv.clone()
    bias_before = bias.clone()

    _transform_bias_rescale_qkv(qkv, bias, 4)

    assert torch.equal(qkv, qkv_before)
    assert torch.equal(bias, bias_before)


def test_transform_bias_rescale_qkv_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._transform_bias_rescale_qkv")

    launches = []
    real_run = mod._transform_bias_rescale_qkv_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._transform_bias_rescale_qkv_kernel, "run", spy_run)
    qkv, bias = make_inputs(2, 5, 4, 2)
    q, k, v = mod._transform_bias_rescale_qkv(qkv, bias, 4)

    assert len(launches) == 1
    grid = launches[0]
    if callable(grid):
        grid = grid({"BLOCK_SIZE": 1024})
    assert tuple(grid) == (1,)  # cdiv(80 elements, BLOCK_SIZE=1024)
    rq, rk, rv = torch_tbq(qkv, bias, 4)
    torch.testing.assert_close(q, rq, rtol=0, atol=0)
