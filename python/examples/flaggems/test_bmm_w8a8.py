import importlib

import pytest
import torch

from .bmm_w8a8 import bmm_w8a8

# oracle: formula per upstream bmm_w8a8.py (no aten op): int8 x int8
# accumulates exactly in int32 (kernel line 54-58), then val = acc.float() *
# sa[:, None] * sb[None, :] before the out_dtype cast (lines 61-63). The
# int32 accumulator is exactly representable in fp32 for these shapes, so
# the pinned comparison is exact after the final cast.


def _ref(activation, weight, sa, sb, out_dtype):
    acc = torch.bmm(activation.long(), weight.long()).float()
    return (acc * sa[:, :, None] * sb[:, None, :]).to(out_dtype)


@pytest.mark.parametrize("batch, M, K, N", [(2, 33, 65, 129), (4, 64, 128, 256)])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bmm_w8a8_matches_formula(batch, M, K, N, out_dtype):
    torch.manual_seed(0)
    activation = torch.randint(-128, 128, (batch, M, K), dtype=torch.int8)
    weight = torch.randint(-128, 128, (batch, N, K), dtype=torch.int8).transpose(1, 2)
    sa = torch.rand(batch, M) * 0.005
    sb = torch.rand(batch, N) * 0.005

    out = bmm_w8a8(activation, weight, sa, sb, out_dtype=out_dtype)
    ref = _ref(activation, weight, sa, sb, out_dtype)
    assert out.shape == (batch, M, N) and out.dtype == out_dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_bmm_w8a8_default_out_dtype_bfloat16():
    torch.manual_seed(1)
    activation = torch.randint(-128, 128, (2, 17, 37), dtype=torch.int8)
    weight = torch.randint(-128, 128, (2, 65, 37), dtype=torch.int8).transpose(1, 2)
    sa = torch.rand(2, 17) * 0.005
    sb = torch.rand(2, 65) * 0.005
    out = bmm_w8a8(activation, weight, sa, sb)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(
        out, _ref(activation, weight, sa, sb, torch.bfloat16), rtol=0, atol=0
    )


def test_bmm_w8a8_rank3_scales_and_out_buffer():
    torch.manual_seed(2)
    batch, M, K, N = 2, 17, 37, 65
    activation = torch.randint(-128, 128, (batch, M, K), dtype=torch.int8)
    weight = torch.randint(-128, 128, (batch, N, K), dtype=torch.int8).transpose(1, 2)
    sa = torch.rand(batch, M) * 0.005
    sb = torch.rand(batch, N) * 0.005
    out = torch.empty((batch, M, N), dtype=torch.float32)

    ret = bmm_w8a8(
        activation,
        weight,
        sa[:, :, None],
        sb[:, None, :],
        out_dtype=torch.float32,
        out=out,
    )
    torch.testing.assert_close(
        out, _ref(activation, weight, sa, sb, torch.float32), rtol=0, atol=0
    )
    assert ret is out or torch.equal(ret, out)


def test_bmm_w8a8_contiguous_weight_layout():
    # Weight handed over row-major (batch, K, N): strides drive the kernel.
    torch.manual_seed(3)
    batch, M, K, N = 3, 16, 32, 24
    activation = torch.randint(-128, 128, (batch, M, K), dtype=torch.int8)
    weight = torch.randint(-128, 128, (batch, K, N), dtype=torch.int8)
    sa = torch.rand(batch, M) * 0.005
    sb = torch.rand(batch, N) * 0.005
    out = bmm_w8a8(activation, weight, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(
        out, _ref(activation, weight, sa, sb, torch.float32), rtol=0, atol=0
    )


def test_bmm_w8a8_negative_scales():
    torch.manual_seed(4)
    batch, M, K, N = 2, 8, 16, 8
    activation = torch.randint(-128, 128, (batch, M, K), dtype=torch.int8)
    weight = torch.randint(-128, 128, (batch, N, K), dtype=torch.int8).transpose(1, 2)
    sa = -torch.rand(batch, M) * 0.005
    sb = torch.rand(batch, N) * 0.005
    out = bmm_w8a8(activation, weight, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(
        out, _ref(activation, weight, sa, sb, torch.float32), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "empty_shape", [(0, 5, 7, 32), (2, 0, 7, 32), (2, 5, 0, 32), (2, 5, 7, 0)]
)
def test_bmm_w8a8_empty(empty_shape):
    batch, M, K, N = empty_shape
    activation = torch.empty((batch, M, K), dtype=torch.int8)
    weight = torch.empty((batch, N, K), dtype=torch.int8).transpose(1, 2)
    sa = torch.ones((batch, M))
    sb = torch.ones((batch, N))
    out = bmm_w8a8(activation, weight, sa, sb, out_dtype=torch.bfloat16)
    assert out.shape == (batch, M, N)
    if out.numel():
        assert torch.all(out == 0)


def test_bmm_w8a8_errors_match_upstream_messages():
    torch.manual_seed(5)
    a = torch.randint(-128, 128, (2, 8, 16), dtype=torch.int8)
    b = torch.randint(-128, 128, (2, 16, 8), dtype=torch.int8)
    sa = torch.ones(2, 8)
    sb = torch.ones(2, 8)
    with pytest.raises(ValueError, match="Expected compatible batched matrices"):
        bmm_w8a8(a, b[:1], sa, sb)
    with pytest.raises(ValueError, match="Expected compatible batched matrices"):
        bmm_w8a8(a, b[:, :8], sa, sb)
    with pytest.raises(TypeError, match="Operands must both have INT8 dtype"):
        bmm_w8a8(a.float(), b, sa, sb)
    with pytest.raises(ValueError, match="Expected per-row A scales"):
        bmm_w8a8(a, b, torch.ones(2, 5), sb)
    with pytest.raises(ValueError, match="per-output-channel B scales"):
        bmm_w8a8(a, b, sa, torch.ones(2, 5))
    with pytest.raises(TypeError, match="Expected FP16, BF16 or FP32 output"):
        bmm_w8a8(a, b, sa, sb, out_dtype=torch.float64)
    with pytest.raises(TypeError, match="Scales must be floating point"):
        bmm_w8a8(a, b, sa.long(), sb)
    with pytest.raises(ValueError, match="Invalid output tensor"):
        bmm_w8a8(
            a,
            b,
            sa,
            sb,
            out=torch.empty((2, 8, 8), dtype=torch.float16),
        )


def test_bmm_w8a8_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.bmm_w8a8")
    launches = []
    kernel = mod.channel_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    batch, M, K, N = 2, 17, 37, 65
    activation = torch.randint(-128, 128, (batch, M, K), dtype=torch.int8)
    weight = torch.randint(-128, 128, (batch, N, K), dtype=torch.int8).transpose(1, 2)
    sa = torch.rand(batch, M) * 0.005
    sb = torch.rand(batch, N) * 0.005
    out = bmm_w8a8(activation, weight, sa, sb, out_dtype=torch.float32)
    assert len(launches) >= 1
    torch.testing.assert_close(
        out, _ref(activation, weight, sa, sb, torch.float32), rtol=0, atol=0
    )
