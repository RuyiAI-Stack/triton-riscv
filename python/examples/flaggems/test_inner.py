import importlib

import pytest
import torch

from .inner import inner


def test_inner_1d_dot_matches_aten():
    torch.manual_seed(0)
    a = torch.randn(64, dtype=torch.float32)
    b = torch.randn(64, dtype=torch.float32)
    out = inner(a, b)
    ref = torch.inner(a, b)
    assert out.shape == ()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_inner_2d_matches_aten_two_shapes():
    torch.manual_seed(1)
    for sa, sb in (((16, 16), (16, 16)), ((33, 17), (5, 17))):
        a = torch.randn(*sa, dtype=torch.float32)
        b = torch.randn(*sb, dtype=torch.float32)
        out = inner(a, b)
        ref = torch.inner(a, b)
        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5, msg=f"{sa}/{sb}")


def test_inner_leading_dims_concatenate_no_broadcast():
    # (2, 3, K) x (5, K) -> (2, 3, 5): leading dims sit per-side.
    torch.manual_seed(2)
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    b = torch.randn(5, 4, dtype=torch.float32)
    out = inner(a, b)
    ref = torch.inner(a, b)
    assert out.shape == (2, 3, 5)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_inner_0d_operand_scales_like_aten():
    torch.manual_seed(3)
    s = torch.tensor(2.5, dtype=torch.float32)
    v = torch.arange(4, dtype=torch.float32)
    out = inner(s, v)
    ref = torch.inner(s, v)
    assert out.shape == (4,)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    out2 = inner(v, s)
    torch.testing.assert_close(out2, torch.inner(v, s), rtol=0, atol=0)
    out3 = inner(torch.tensor(3.0), torch.tensor(-2.0))
    assert out3.shape == ()
    torch.testing.assert_close(
        out3, torch.inner(torch.tensor(3.0), torch.tensor(-2.0)), rtol=0, atol=0
    )


def test_inner_arange_exact_all_float_dtypes():
    # Integer-valued data makes the contraction exact regardless of the
    # kernel's tile order: bit-for-bit equality with aten.
    a = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 4
    b = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4) / 4
    torch.testing.assert_close(inner(a, b), torch.inner(a, b), rtol=0, atol=0)
    a64 = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4) / 4
    b64 = torch.arange(5 * 4, dtype=torch.float64).reshape(5, 4) / 4
    torch.testing.assert_close(inner(a64, b64), torch.inner(a64, b64), rtol=0, atol=0)


def test_inner_fp64_randn():
    # Both sides accumulate in fp64; only the summation order differs.
    # rtol/atol 1e-9 measured against aten fp64 randn.
    torch.manual_seed(4)
    a = torch.randn(8, 8, dtype=torch.float64)
    b = torch.randn(6, 8, dtype=torch.float64)
    out = inner(a, b)
    torch.testing.assert_close(out, torch.inner(a, b), rtol=1e-9, atol=1e-9)
    v1 = torch.randn(128, dtype=torch.float64)
    v2 = torch.randn(128, dtype=torch.float64)
    torch.testing.assert_close(inner(v1, v2), torch.inner(v1, v2), rtol=1e-9, atol=1e-9)


def test_inner_int64_small_values_exact():
    a = torch.arange(2 * 4, dtype=torch.int64).reshape(2, 4) % 5
    b = torch.arange(3 * 4, dtype=torch.int64).reshape(3, 4) % 4
    out = inner(a, b)
    ref = torch.inner(a, b)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.int64


def test_inner_zero_k_is_zeros_like_aten():
    a = torch.empty(2, 3, 0, dtype=torch.float32)
    b = torch.empty(5, 0, dtype=torch.float32)
    out = inner(a, b)
    ref = torch.inner(a, b)
    assert out.shape == (2, 3, 5)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_inner_empty_leading_dim():
    torch.manual_seed(5)
    a = torch.empty(0, 4, dtype=torch.float32)
    b = torch.randn(5, 4, dtype=torch.float32)
    out = inner(a, b)
    ref = torch.inner(a, b)
    assert out.shape == (0, 5)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_inner_non_contiguous_input():
    torch.manual_seed(6)
    base = torch.randn(2, 3, dtype=torch.float32)
    a = base.t()  # (3, 2) non-contiguous view
    assert not a.is_contiguous()
    b = torch.randn(5, 2, dtype=torch.float32)
    out = inner(a, b)
    ref = torch.inner(a, b)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_inner_errors_match_aten():
    torch.manual_seed(7)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 4, dtype=torch.float32)
    with pytest.raises(
        RuntimeError,
        match=r"inner\(\) the last dimension must match on both input tensors",
    ):
        inner(a, b)
    with pytest.raises(RuntimeError, match="both inputs should have same dtype"):
        inner(a, b.to(torch.float64))


def test_inner_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.inner")
    dot_launches = []
    gemm_launches = []
    dot_kernel = mod._inner_dot_kernel  # kernel name per upstream
    gemm_kernel = mod._inner_gemm_kernel  # kernel name per upstream
    real_dot = dot_kernel.run
    real_gemm = gemm_kernel.run

    def spy_dot(*args, _real=real_dot, **kwargs):
        dot_launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    def spy_gemm(*args, _real=real_gemm, **kwargs):
        gemm_launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(dot_kernel, "run", spy_dot)
    monkeypatch.setattr(gemm_kernel, "run", spy_gemm)
    torch.manual_seed(8)
    v1 = torch.randn(64, dtype=torch.float32)
    v2 = torch.randn(64, dtype=torch.float32)
    out1 = inner(v1, v2)
    assert len(dot_launches) >= 1
    torch.testing.assert_close(out1, torch.inner(v1, v2), rtol=1e-5, atol=1e-5)
    a = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(4, 16, dtype=torch.float32)
    out2 = inner(a, b)
    assert len(gemm_launches) >= 1
    torch.testing.assert_close(out2, torch.inner(a, b), rtol=1e-5, atol=1e-5)
