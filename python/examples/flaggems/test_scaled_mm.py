import importlib

import pytest
import torch

from .scaled_mm import scaled_mm, scaled_mm_out

# bf16 outputs round at ~4e-3 relative (scales stay fp32); atol/rtol
# 1e-2 covers the output cast and accumulation order.


def _fp8(shape, dtype=torch.float8_e4m3fn):
    # fp8 inputs are constructed by casting known floats; the oracle consumes
    # the SAME casted tensors so the decode is exact on both sides.
    return (torch.randn(shape, dtype=torch.float32) * 0.25).to(dtype)


def _mat2_colmajor(k, n, dtype=torch.float8_e4m3fn):
    # aten._scaled_mm wants mat2 in column-major (the transposed layout).
    return (torch.randn((n, k), dtype=torch.float32) * 0.25).to(dtype).t()


@pytest.mark.parametrize("M, N, K", [(32, 48, 64), (64, 96, 128), (17, 65, 48)])
def test_scaled_mm_tensorwise_matches_aten(M, N, K):
    torch.manual_seed(0)
    a = _fp8((M, K))
    b = _mat2_colmajor(K, N)
    scale_a = torch.tensor(0.75)
    scale_b = torch.tensor(1.25)

    out = scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32)
    ref = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32)
    assert out.shape == (M, N) and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("M, N, K", [(32, 48, 64), (48, 32, 96)])
def test_scaled_mm_rowwise_matches_aten(M, N, K):
    torch.manual_seed(1)
    a = _fp8((M, K))
    b = _mat2_colmajor(K, N)
    scale_a = torch.rand(M, 1) + 0.5
    scale_b = torch.rand(1, N) + 0.5

    out = scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)
    ref = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_scaled_mm_default_out_dtype_is_input_dtype():
    # Upstream _resolve_out_dtype: out_dtype None and out None -> self.dtype.
    torch.manual_seed(2)
    a = _fp8((16, 32))
    b = _mat2_colmajor(32, 24)
    out = scaled_mm(a, b, torch.tensor(1.0), torch.tensor(1.0))
    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == (16, 24)


def test_scaled_mm_e5m2_matches_aten():
    torch.manual_seed(3)
    a = _fp8((32, 64), dtype=torch.float8_e5m2)
    b = _mat2_colmajor(64, 48, dtype=torch.float8_e5m2)
    out = scaled_mm(a, b, torch.tensor(1.0), torch.tensor(2.0), out_dtype=torch.float32)
    ref = torch._scaled_mm(
        a, b, torch.tensor(1.0), torch.tensor(2.0), out_dtype=torch.float32
    )
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_scaled_mm_bias_bfloat16_matches_aten():
    torch.manual_seed(4)
    M, N, K = 32, 48, 64
    a = _fp8((M, K))
    b = _mat2_colmajor(K, N)
    bias = torch.randn(N, dtype=torch.bfloat16)

    out = scaled_mm(
        a, b, torch.tensor(1.0), torch.tensor(1.0), bias=bias, out_dtype=torch.bfloat16
    )
    ref = torch._scaled_mm(
        a, b, torch.tensor(1.0), torch.tensor(1.0), bias=bias, out_dtype=torch.bfloat16
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_scaled_mm_scale_result_is_mathematically_ignored():
    # Upstream comment: scale_result does not change the result (ATen GPU
    # semantics kept by this implementation).
    torch.manual_seed(5)
    a = _fp8((16, 32))
    b = _mat2_colmajor(32, 24)
    plain = scaled_mm(
        a, b, torch.tensor(1.0), torch.tensor(1.0), out_dtype=torch.float32
    )
    scaled = scaled_mm(
        a,
        b,
        torch.tensor(1.0),
        torch.tensor(1.0),
        scale_result=torch.tensor(3.0),
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(plain, scaled, rtol=0, atol=0)


def test_scaled_mm_zero_and_negative_scales():
    torch.manual_seed(6)
    a = _fp8((16, 32))
    b = _mat2_colmajor(32, 24)
    zeros = scaled_mm(
        a, b, torch.tensor(0.0), torch.tensor(2.0), out_dtype=torch.float32
    )
    assert torch.all(zeros == 0)

    pos = scaled_mm(a, b, torch.tensor(1.0), torch.tensor(1.0), out_dtype=torch.float32)
    neg = scaled_mm(
        a, b, torch.tensor(-1.0), torch.tensor(1.0), out_dtype=torch.float32
    )
    torch.testing.assert_close(neg, -pos, rtol=0, atol=0)


def test_scaled_mm_zero_rows_or_cols():
    torch.manual_seed(7)
    a = _fp8((0, 32))
    b = _mat2_colmajor(32, 24)
    out = scaled_mm(a, b, torch.tensor(1.0), torch.tensor(1.0))
    assert out.shape == (0, 24)

    a2 = _fp8((16, 32))
    b2 = _mat2_colmajor(32, 0)
    out2 = scaled_mm(a2, b2, torch.tensor(1.0), torch.tensor(1.0))
    assert out2.shape == (16, 0)


def test_scaled_mm_out_writes_preallocated_buffer():
    torch.manual_seed(8)
    M, N, K = 32, 48, 64
    a = _fp8((M, K))
    b = _mat2_colmajor(K, N)
    out = torch.empty((M, N), dtype=torch.float32)
    ret = scaled_mm_out(
        a, b, torch.tensor(1.0), torch.tensor(1.0), out_dtype=torch.float32, out=out
    )
    ref = torch._scaled_mm(
        a, b, torch.tensor(1.0), torch.tensor(1.0), out_dtype=torch.float32
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    assert ret is out or torch.equal(ret, out)


def test_scaled_mm_out_resizes_mismatched_buffer():
    torch.manual_seed(9)
    M, N, K = 16, 24, 32
    a = _fp8((M, K))
    b = _mat2_colmajor(K, N)
    out = torch.empty((4, 5), dtype=torch.float32)
    ret = scaled_mm_out(
        a, b, torch.tensor(1.0), torch.tensor(1.0), out_dtype=torch.float32, out=out
    )
    assert ret.shape == (M, N)
    ref = torch._scaled_mm(
        a, b, torch.tensor(1.0), torch.tensor(1.0), out_dtype=torch.float32
    )
    torch.testing.assert_close(ret, ref, rtol=1e-4, atol=1e-4)


def test_scaled_mm_errors_match_upstream_messages():
    torch.manual_seed(10)
    a = _fp8((16, 32))
    b = _mat2_colmajor(32, 24)
    with pytest.raises(RuntimeError, match="mat1 and mat2 must be matrices"):
        scaled_mm(a[0], b, torch.tensor(1.0), torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
        scaled_mm(a, _mat2_colmajor(16, 24), torch.tensor(1.0), torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="mat1 and mat2 must be Float8 matrices"):
        scaled_mm(a.float(), b, torch.tensor(1.0), torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="scale_a must be Float32"):
        scaled_mm(a, b, torch.tensor(1.0, dtype=torch.float64), torch.tensor(1.0))
    with pytest.raises(
        RuntimeError, match="scale_a must be a scalar or a contiguous tensor"
    ):
        scaled_mm(a, b, torch.rand(4, 4), torch.tensor(1.0))
    with pytest.raises(
        RuntimeError, match="scale_b must be a scalar or a contiguous tensor"
    ):
        scaled_mm(a, b, torch.tensor(1.0), torch.rand(1, 7))
    with pytest.raises(RuntimeError, match="Bias must be size"):
        scaled_mm(
            a,
            b,
            torch.tensor(1.0),
            torch.tensor(1.0),
            bias=torch.zeros(5, dtype=torch.bfloat16),
        )
    with pytest.raises(RuntimeError, match="Bias must be BFloat16 or Half"):
        scaled_mm(a, b, torch.tensor(1.0), torch.tensor(1.0), bias=torch.zeros(24))
    with pytest.raises(
        RuntimeError, match="Bias is not supported when out_dtype is set to Float32"
    ):
        scaled_mm(
            a,
            b,
            torch.tensor(1.0),
            torch.tensor(1.0),
            bias=torch.zeros(24, dtype=torch.bfloat16),
            out_dtype=torch.float32,
        )
    with pytest.raises(RuntimeError, match="out_dtype must match output matrix type"):
        scaled_mm_out(
            a,
            b,
            torch.tensor(1.0),
            torch.tensor(1.0),
            out_dtype=torch.float32,
            out=torch.empty((16, 24), dtype=torch.bfloat16),
        )
    with pytest.raises(
        RuntimeError, match="out_dtype must be Float8, Float16, BFloat16 or Float32"
    ):
        scaled_mm(a, b, torch.tensor(1.0), torch.tensor(1.0), out_dtype=torch.int64)


def test_scaled_mm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.scaled_mm")
    launches = []
    kernel = mod.scaled_mm_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    a = _fp8((32, 64))
    b = _mat2_colmajor(64, 48)
    out = scaled_mm(a, b, torch.tensor(1.0), torch.tensor(1.0), out_dtype=torch.float32)
    ref = torch._scaled_mm(
        a, b, torch.tensor(1.0), torch.tensor(1.0), out_dtype=torch.float32
    )
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
