import math

import pytest
import torch

from .fmod_ import fmod, fmod_

# Bit-exactness (probed against torch CPU): both sides reduce to IEEE frem
# for floats -- frem is exact, and the fp16 path (widened fp32 opmath +
# final rounding) matches ATen's own opmath -- and to exact integer
# arithmetic for ints, so every test below compares with rtol=0, atol=0.


def _float_pair(size, dtype, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(size, dtype=dtype) * 10.0
    y = torch.randn(size, dtype=dtype) * 5.0 + 1.0
    return x, y


def _int_pair(size, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(-1000, 1000, (size,), dtype=torch.int64)
    y = torch.randint(-100, 100, (size,), dtype=torch.int64)
    y[y == 0] = 7
    return x, y


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_fmod_tt(dtype, size):
    x, y = _float_pair(size, dtype)

    ref = torch.fmod(x, y)
    tri = fmod(x, y)

    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_fmod_integer_tt(size):
    # Integer fmod is bit-exact (sign follows the dividend).
    x, y = _int_pair(size)

    ref = torch.fmod(x, y)
    tri = fmod(x, y)

    assert tri.dtype == torch.int64
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_fmod_integer_min():
    # INT_MIN % -1: torch folds to 0; the trunc-division formula reaches
    # the same 0 through two's-complement wraparound.
    i64min = torch.iinfo(torch.int64).min
    i32min = torch.iinfo(torch.int32).min
    x = torch.tensor([i64min, i64min, i64min, 7, -7], dtype=torch.int64)
    y = torch.tensor([-1, 1, 2, 3, 3], dtype=torch.int64)

    ref = torch.fmod(x, y)
    tri = fmod(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri[0].item() == 0 and tri[1].item() == 0

    x32 = torch.tensor([i32min, i32min], dtype=torch.int32)
    y32 = torch.tensor([-1, 1], dtype=torch.int32)

    tri32 = fmod(x32, y32)

    torch.testing.assert_close(tri32, torch.fmod(x32, y32), rtol=0, atol=0)
    assert tri32[0].item() == 0 and tri32[1].item() == 0


@pytest.mark.parametrize("size", [512, 1023])
def test_fmod_tensor_scalar(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 10.0

    ref = torch.fmod(x, 2.5)
    tri = fmod(x, 2.5)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("size", [512, 1023])
def test_fmod_tensor_scalar_int(size):
    x, _ = _int_pair(size)

    ref = torch.fmod(x, 7)
    tri = fmod(x, 7)

    assert tri.dtype == torch.int64
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("size", [512, 1023])
def test_fmod_scalar_tensor(size):
    torch.manual_seed(0)
    y = torch.randn(size, dtype=torch.float32) * 5.0 + 1.0

    # torch.fmod rejects a scalar first argument; the reference uses the
    # equivalent 0-dim tensor (same values ATen would see).
    ref = torch.fmod(torch.tensor(-2.5), y)
    tri = fmod(-2.5, y)

    assert tri.dtype == torch.float32
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_fmod_special_values():
    # IEEE specials, probed on torch CPU fp32: fmod keeps the sign of the
    # dividend; |x| finite next to +-inf returns x; any inf or NaN operand
    # yields NaN; a float zero divisor yields NaN.
    pos0, neg0 = 0.0, -0.0
    inf, ninf, nan = float("inf"), float("-inf"), float("nan")
    x = torch.tensor([neg0, 5.0, 5.0, inf, ninf, inf, nan, 3.0, -3.0, 7.0, -7.0])
    y = torch.tensor([3.0, inf, ninf, 3.0, 3.0, inf, 3.0, pos0, pos0, 3.0, 3.0])

    ref = torch.fmod(x, y)
    tri = fmod(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert math.copysign(1.0, tri[0].item()) < 0  # fmod(-0, 3) == -0
    assert tri[1].item() == 5.0  # fmod(5, inf) == 5
    assert tri[2].item() == 5.0  # fmod(5, -inf) == 5
    assert tri[9].item() == 1.0 and tri[10].item() == -1.0  # sign of x


def test_fmod_integer_zero_divisor():
    # torch raises RuntimeError('ZeroDivisionError') for integer zero
    # divisors (a raw hardware divide would kill the process instead).
    x, y = _int_pair(16)
    y[3] = 0

    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        fmod(x, y)
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        fmod(x, 0)


def test_fmod_float_zero_divisor_is_nan():
    x = torch.tensor([3.0, -3.0, 0.0])
    y = torch.tensor([0.0, 0.0, 0.0])

    tri = fmod(x, y)

    torch.testing.assert_close(tri, torch.fmod(x, y), rtol=0, atol=0, equal_nan=True)


def test_fmod_empty():
    x = torch.empty(0, dtype=torch.float32)
    y = torch.empty(0, dtype=torch.float32)

    tri = fmod(x, y)

    assert tri.shape == (0,)
    torch.testing.assert_close(tri, torch.fmod(x, y), rtol=0, atol=0)


@pytest.mark.parametrize("shape_x, shape_y", [((16, 256), (256,)), ((4, 1), (4, 128))])
def test_fmod_broadcast(shape_x, shape_y):
    torch.manual_seed(0)
    x = torch.randn(shape_x, dtype=torch.float32) * 10.0
    y = torch.randn(shape_y, dtype=torch.float32) * 5.0 + 1.0

    tri = fmod(x, y)

    assert tri.shape == torch.fmod(x, y).shape
    torch.testing.assert_close(tri, torch.fmod(x, y), rtol=0, atol=0)


def test_fmod_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.float32).t() * 10.0
    y = torch.randn(32, 64, dtype=torch.float32).t() * 5.0 + 1.0
    assert not x.is_contiguous() and not y.is_contiguous()

    tri = fmod(x, y)

    torch.testing.assert_close(tri, torch.fmod(x, y), rtol=0, atol=0)


def test_fmod_dtype_promotion():
    torch.manual_seed(0)
    x32 = torch.randn(64, dtype=torch.float32) * 10.0
    y64 = torch.randn(64, dtype=torch.float64) * 5.0 + 1.0
    tri = fmod(x32, y64)
    assert tri.dtype == torch.float64
    torch.testing.assert_close(tri, torch.fmod(x32, y64), rtol=0, atol=0)

    xi = torch.randint(-100, 100, (64,), dtype=torch.int32)
    yj = torch.randint(-50, 50, (64,), dtype=torch.int64)
    yj[yj == 0] = 3
    tri = fmod(xi, yj)
    assert tri.dtype == torch.int64
    torch.testing.assert_close(tri, torch.fmod(xi, yj), rtol=0, atol=0)

    # Weak scalars: a fp16 tensor keeps fp16; an int tensor next to a
    # float scalar promotes to the default float dtype.
    x16 = torch.randn(64, dtype=torch.float16) * 10.0
    assert fmod(x16, 2.5).dtype == torch.float16
    xint, _ = _int_pair(64)
    assert fmod(xint, 2.5).dtype == torch.float32


def test_fmod_both_scalar():
    # torch.fmod itself rejects two scalars; the fallback here must keep
    # C semantics (sign of the dividend).  Upstream computes A % B, whose
    # sign follows the divisor -- a divergence pinned by this test.
    assert fmod(7.0, 2.0).item() == 1.0
    assert fmod(-7.0, 2.0).item() == -1.0


@pytest.mark.parametrize("size", [512, 1024])
def test_fmod__inplace(size):
    x, y = _float_pair(size, torch.float32)
    x_ref = x.clone()

    x_ref.fmod_(y)
    ret = fmod_(x, y)

    assert ret is x
    torch.testing.assert_close(x, x_ref, rtol=0, atol=0)


def test_fmod__inplace_scalar():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32) * 10.0
    x_ref = x.clone()

    x_ref.fmod_(2.5)
    ret = fmod_(x, 2.5)

    assert ret is x
    torch.testing.assert_close(x, x_ref, rtol=0, atol=0)


def test_fmod__inplace_fp64():
    x, y = _float_pair(512, torch.float64)
    x_ref = x.clone()

    x_ref.fmod_(y)
    ret = fmod_(x, y)

    assert ret is x
    torch.testing.assert_close(x, x_ref, rtol=0, atol=0)


def test_fmod__inplace_non_contiguous():
    # Write-through: the in-place result must land in the base storage,
    # including for a genuinely non-contiguous view.
    torch.manual_seed(0)
    base = torch.randn(32, 64, dtype=torch.float32) * 10.0
    x = base[::2, ::2].t()
    y = torch.randn(x.shape, dtype=torch.float32) * 5.0 + 1.0
    assert not x.is_contiguous()

    x_ref = x.clone()
    x_ref.fmod_(y)

    ret = fmod_(x, y)

    assert ret is x
    torch.testing.assert_close(x, x_ref, rtol=0, atol=0)
    torch.testing.assert_close(base[::2, ::2].t(), x_ref, rtol=0, atol=0)


def test_fmod__inplace_empty():
    x = torch.empty(0, dtype=torch.float32)
    y = torch.empty(0, dtype=torch.float32)

    ret = fmod_(x, y)

    assert ret is x
    assert x.numel() == 0


def test_fmod__rejects_integer():
    # Upstream restricts fmod_ to floating point dtypes (ATen itself also
    # serves integers -- a deliberate upstream narrowing, mirrored here).
    x, y = _int_pair(16)

    with pytest.raises(AssertionError, match="floating point"):
        fmod_(x, y)


def test_fmod__inplace_float_zero_divisor():
    # In-place on data containing float zero divisors stays on the NaN
    # contract (no integer-style error, no crash).
    x, y = _int_pair(16)
    y[3] = 0

    x_f = x.float()
    tri = fmod_(x_f, y.float())
    assert tri is x_f
    torch.testing.assert_close(
        x_f, torch.fmod(x.float(), y.float()), rtol=0, atol=0, equal_nan=True
    )


def test_fmod_triton_path(monkeypatch):
    # Smoke: the fp32 main path must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.fmod_")

    launches = []
    real_run = mod.fmod_kernel_tt.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.fmod_kernel_tt, "run", spy_run)
    x, y = _float_pair(1024, torch.float32)
    out = mod.fmod(x, y)
    assert len(launches) == 1, "fmod(x, y) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.fmod(x, y), rtol=0, atol=0)
