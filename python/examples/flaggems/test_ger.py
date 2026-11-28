import importlib

import pytest
import torch

from .ger import ger


def test_ger_matches_aten_two_shapes():
    # ger is a single elementwise multiply per output, so it must match the
    # aten oracle bit for bit in every dtype.
    torch.manual_seed(0)
    for m, n in ((3, 4), (17, 33)):
        a = torch.randn(m, dtype=torch.float32)
        b = torch.randn(n, dtype=torch.float32)
        out = ger(a, b)
        ref = torch.ger(a, b)  # deprecated oracle, warning is fine
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"{(m, n)}")
        assert out.shape == (m, n) and out.dtype == torch.float32


def test_ger_fp64_exact():
    torch.manual_seed(1)
    a = torch.randn(8, dtype=torch.float64)
    b = torch.randn(5, dtype=torch.float64)
    out = ger(a, b)
    ref = torch.ger(a, b)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_ger_int64_exact():
    a = torch.arange(1, 6, dtype=torch.int64)
    b = torch.arange(2, 9, dtype=torch.int64)
    out = ger(a, b)
    ref = torch.ger(a, b)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.int64


def test_ger_complex64_exact():
    torch.manual_seed(2)
    a = torch.view_as_complex(torch.randn(4, 2))
    b = torch.view_as_complex(torch.randn(6, 2))
    out = ger(a, b)
    ref = torch.ger(a, b)
    # The aten oracle's complex multiply is a mix of its vectorized kernel
    # (mul/mul/hsub, round-per-op) and scalar FMA-contracted elements, so its
    # bits are not reproducible by any single elementwise formula; compare at
    # the 1-2 ulp level of the unit-scale products instead (measured max diff
    # 6.0e-08 on this seed).
    torch.testing.assert_close(out, ref, rtol=1e-7, atol=1e-7)
    assert out.dtype == torch.complex64


def test_ger_out_variant_writes_buffer():
    torch.manual_seed(3)
    a = torch.randn(4, dtype=torch.float32)
    b = torch.randn(3, dtype=torch.float32)
    out = torch.empty(4, 3, dtype=torch.float32)
    ret = ger(a, b, out=out)
    assert ret is out
    torch.testing.assert_close(out, torch.ger(a, b), rtol=0, atol=0)


def test_ger_out_positional_writes_buffer():
    torch.manual_seed(4)
    a = torch.randn(4, dtype=torch.float64)
    b = torch.randn(3, dtype=torch.float64)
    out = torch.empty(4, 3, dtype=torch.float64)
    ret = ger(a, b, out)
    assert ret is out
    torch.testing.assert_close(out, torch.ger(a, b), rtol=0, atol=0)


def test_ger_empty_operands():
    torch.manual_seed(5)
    a = torch.empty(0, dtype=torch.float32)
    b = torch.randn(3, dtype=torch.float32)
    out = ger(a, b)
    ref = torch.ger(a, b)
    assert out.shape == (0, 3)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    out2 = ger(torch.randn(3), torch.empty(0))
    assert out2.shape == (3, 0)


def test_ger_non_contiguous_strided_view():
    torch.manual_seed(6)
    base = torch.randn(8, dtype=torch.float64)
    a = base[::2]  # stride-2 view
    assert not a.is_contiguous()
    b = torch.randn(5, dtype=torch.float64)
    out = ger(a, b)
    torch.testing.assert_close(out, torch.ger(a, b), rtol=0, atol=0)


def test_ger_errors_match_upstream_validation():
    torch.manual_seed(7)
    a = torch.randn(4, dtype=torch.float32)
    b = torch.randn(3, dtype=torch.float32)
    with pytest.raises(ValueError, match="ger: Expected 1-D argument self"):
        ger(torch.randn(2, 3), b)
    with pytest.raises(ValueError, match="ger: Expected 1-D argument vec2"):
        ger(a, torch.randn(2, 3))
    with pytest.raises(RuntimeError, match="fp8 operands are not supported"):
        ger(
            torch.randn(4).to(torch.float8_e4m3fn),
            torch.randn(3).to(torch.float8_e4m3fn),
        )
    with pytest.raises(TypeError, match="ger: expected Tensor arguments"):
        ger([1.0, 2.0], b)


def test_ger_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.ger")
    launches = []
    kernel = mod._ger_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(8)
    a = torch.arange(1, 8, dtype=torch.float64)
    b = torch.arange(2, 9, dtype=torch.float64)
    out = ger(a, b)
    assert len(launches) >= 1
    torch.testing.assert_close(out, torch.ger(a, b), rtol=0, atol=0)
