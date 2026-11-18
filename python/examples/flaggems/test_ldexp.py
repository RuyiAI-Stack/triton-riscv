import struct

import pytest
import torch

from .ldexp import ldexp, ldexp_out


def _bits(t):
    return struct.unpack("I", struct.pack("f", t.item()))[0]


def _assert_ldexp_bitexact(x, e):
    ref = torch.ldexp(x, e)
    out = ldexp(x, e)
    assert out.dtype == ref.dtype
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_ldexp_random(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * torch.pow(
        2.0, torch.randint(-60, 60, (size,)).float()
    )
    e = torch.randint(-100, 100, (size,), dtype=torch.int32)
    _assert_ldexp_bitexact(x, e)


@pytest.mark.parametrize("e_dtype", [torch.int16, torch.int32, torch.int64])
def test_ldexp_exponent_overflow_underflow(e_dtype):
    # e near +-2**31 saturates the result to +-inf / signed zero, and torch
    # 2.11+ wraps out-of-int32 int64 exponents (2**32 + 5 acts as 5).
    if e_dtype == torch.int16:
        raw = [30000, 30000, -30000, -30000, 5]
    else:
        raw = [2**31 - 1, 2**31 - 1, -(2**31), -(2**31), 2**32 + 5]
    x = torch.tensor([1.5, -1.5, 0.75, -0.75, 6e-8], dtype=torch.float32)
    e = torch.tensor(raw, dtype=torch.int64).to(e_dtype)
    _assert_ldexp_bitexact(x, e)
    if e_dtype != torch.int16:
        assert torch.ldexp(x, e)[4] == ldexp(x, e)[4] == 6e-8 * 32.0


def test_ldexp_special_values():
    values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        1.5,
        -2.25,
        1.0 + 2.0**-23,
        2.0**126,
        2.0**127,
        torch.finfo(torch.float32).max,
        -2.0**126,
        2.0**-126,
        2.0**-149,
        -(2.0**-149),
        6e-8,
        -6e-8,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    exps = [0, 1, -1, 5, -5, 24, -24, 25, -25, 26, -26, 127, -127, 128, -128, 149, -149, 150, -150, 300, -300]
    x = torch.tensor(values * len(exps), dtype=torch.float32)
    e = torch.tensor(exps * len(values), dtype=torch.int32)
    _assert_ldexp_bitexact(x, e)


def test_ldexp_subnormal_single_rounding():
    # scalbn rounds once: (1 + 3*2**-23) * 2**-1 * 2**-127 = 2**-128 +
    # 0.75 ulp rounds up to 0x200001.  A x*exp2(e) implementation double
    # rounds this to 0x200000.
    x = torch.tensor([(1.0 + 3 * 2.0**-23) * 0.5], dtype=torch.float32)
    e = torch.tensor([-127], dtype=torch.int32)
    out = ldexp(x, e)
    assert _bits(out) == 0x200001
    _assert_ldexp_bitexact(x, e)


@pytest.mark.parametrize("x_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("e_dtype", [torch.int16, torch.int32, torch.int64])
def test_ldexp_dtype_matrix(x_dtype, e_dtype):
    torch.manual_seed(0)
    x = (
        torch.randn(512, dtype=torch.float32)
        * torch.pow(2.0, torch.randint(-12, 12, (512,)).float())
    ).to(x_dtype)
    e = torch.randint(-30, 30, (512,), dtype=torch.int64).to(e_dtype)
    _assert_ldexp_bitexact(x, e)


@pytest.mark.parametrize("x_dtype", [torch.int32, torch.int64])
def test_ldexp_integer_input_promotes_float(x_dtype):
    # INT_TO_FLOAT promotion: integer base yields a float32 result.
    torch.manual_seed(0)
    x = torch.randint(-1000, 1000, (512,), dtype=torch.int64).to(x_dtype)
    e = torch.randint(-30, 30, (512,), dtype=torch.int64)
    _assert_ldexp_bitexact(x, e)


def test_ldexp_float_exponent():
    # torch computes a float exponent as x * exp2(e) in float32.
    x = torch.tensor([1.5, -2.25, 0.5, 8.0], dtype=torch.float32)
    e = torch.tensor([0.5, 1.5, -0.5, -2.5], dtype=torch.float32)
    _assert_ldexp_bitexact(x, e)

    torch.manual_seed(0)
    xr = torch.randn(512, dtype=torch.float32)
    er = torch.randn(512, dtype=torch.float32) * 3
    ref = torch.ldexp(xr, er)
    out = ldexp(xr, er)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


def test_ldexp_float64_fallback():
    # float64 needs a double-precision pipeline; it delegates to torch.
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float64)
    e = torch.randint(-40, 40, (128,), dtype=torch.int32)
    ref = torch.ldexp(x, e)
    out = ldexp(x, e)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


def test_ldexp_broadcast():
    torch.manual_seed(0)
    x = torch.randn(2, 1, dtype=torch.float32)
    e = torch.randint(-5, 5, (1, 4), dtype=torch.int32)
    _assert_ldexp_bitexact(x, e)


@pytest.mark.parametrize("shape", [(0,), (0, 37)])
def test_ldexp_empty(shape):
    x = torch.randn(shape, dtype=torch.float32)
    e = torch.randint(-5, 5, shape, dtype=torch.int32)
    out = ldexp(x, e)
    assert out.shape == shape
    assert out.dtype == torch.float32


def test_ldexp_non_contiguous():
    torch.manual_seed(0)
    base = torch.randn(64, 32, dtype=torch.float32)
    x = base.t()
    assert not x.is_contiguous()
    e = torch.randint(-20, 20, (32, 64), dtype=torch.int32)
    _assert_ldexp_bitexact(x, e)


def test_ldexp_scalar_other_raises():
    x = torch.randn(4, dtype=torch.float32)
    with pytest.raises(TypeError):
        ldexp(x, 3)
    with pytest.raises(TypeError):
        ldexp(x, 1.5)


def test_ldexp_out():
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32)
    e = torch.randint(-10, 10, (128,), dtype=torch.int32)
    ref = torch.ldexp(x, e)

    out = torch.empty(128, dtype=torch.float32)
    ret = ldexp_out(x, e, out=out)
    assert ret is out
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)

    wrong = torch.empty(128, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
        ldexp_out(x, e, out=wrong)


def test_ldexp_triton_path(monkeypatch):
    # Smoke: the float32 + int32 path must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.ldexp")

    launches = []
    real_run = mod.ldexp_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.ldexp_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    e = torch.randint(-10, 10, (1024,), dtype=torch.int32)
    ref = torch.ldexp(x, e)

    out = mod.ldexp(x, e)
    assert len(launches) == 1, "ldexp(A, B) must go through the Triton kernel"
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
