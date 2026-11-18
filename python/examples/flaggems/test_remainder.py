import math

import pytest
import torch

from .remainder import remainder, remainder_

# Bit-exactness (probed against torch CPU): both sides reduce to IEEE frem
# plus one exact r + y adjustment for floats (fp16 via ATen's fp32 opmath)
# and to exact integer arithmetic for ints, so every comparison below uses
# rtol=0, atol=0.


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
def test_remainder_tt(dtype, size):
    x, y = _float_pair(size, dtype)

    ref = torch.remainder(x, y)
    tri = remainder(x, y)

    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_remainder_integer_tt(size):
    x, y = _int_pair(size)

    ref = torch.remainder(x, y)
    tri = remainder(x, y)

    assert tri.dtype == torch.int64
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_remainder_sign_follows_divisor():
    # Python-style semantics: the sign of the result follows the divisor.
    x = torch.tensor([7, -7, 7, -7], dtype=torch.int64)
    y = torch.tensor([3, 3, -3, -3], dtype=torch.int64)

    tri = remainder(x, y)

    assert tri.tolist() == [1, 2, -2, -1]


def test_remainder_integer_min():
    i64min = torch.iinfo(torch.int64).min
    i32min = torch.iinfo(torch.int32).min
    x = torch.tensor([i64min, i64min, i64min, 7, -7], dtype=torch.int64)
    y = torch.tensor([-1, 1, 2, 3, 3], dtype=torch.int64)

    ref = torch.remainder(x, y)
    tri = remainder(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri[0].item() == 0 and tri[1].item() == 0

    x32 = torch.tensor([i32min, i32min], dtype=torch.int32)
    y32 = torch.tensor([-1, 1], dtype=torch.int32)

    tri32 = remainder(x32, y32)

    torch.testing.assert_close(tri32, torch.remainder(x32, y32), rtol=0, atol=0)
    assert tri32[0].item() == 0 and tri32[1].item() == 0


@pytest.mark.parametrize("size", [512, 1023])
def test_remainder_tensor_scalar(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 10.0

    ref = torch.remainder(x, 2.5)
    tri = remainder(x, 2.5)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("size", [512, 1023])
def test_remainder_tensor_scalar_negative(size):
    x, _ = _int_pair(size)

    ref = torch.remainder(x, -7)
    tri = remainder(x, -7)

    assert tri.dtype == torch.int64
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("size", [512, 1023])
def test_remainder_scalar_tensor(size):
    torch.manual_seed(0)
    y = torch.randn(size, dtype=torch.float32) * 5.0 + 1.0

    # torch.remainder rejects a scalar first argument; the reference uses
    # the equivalent 0-dim tensor.
    ref = torch.remainder(torch.tensor(-2.5), y)
    tri = remainder(-2.5, y)

    assert tri.dtype == torch.float32
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_remainder_special_values():
    # IEEE specials, probed on torch CPU fp32: the sign follows the
    # divisor, remainder(x, -inf) is -inf for finite positive x, any inf or
    # NaN operand yields NaN, and a float zero divisor yields NaN.
    pos0, neg0 = 0.0, -0.0
    inf, ninf, nan = float("inf"), float("-inf"), float("nan")
    x = torch.tensor([neg0, 5.0, 5.0, inf, 3.0, 7.5, -7.5, 7.5, -7.5, nan])
    y = torch.tensor([3.0, inf, ninf, 3.0, pos0, 2.0, 2.0, -2.0, -2.0, 3.0])

    ref = torch.remainder(x, y)
    tri = remainder(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert math.copysign(1.0, tri[0].item()) < 0  # remainder(-0, 3) == -0
    assert tri[1].item() == 5.0  # remainder(5, inf) == 5
    assert tri[2].item() == float("-inf")  # remainder(5, -inf) == -inf
    assert tri[5].item() == 1.5  # sign of divisor, positive y
    assert tri[6].item() == 0.5
    assert tri[7].item() == -0.5  # sign of divisor, negative y
    assert tri[8].item() == -1.5


def test_remainder_integer_zero_divisor():
    x, y = _int_pair(16)
    y[3] = 0

    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        remainder(x, y)
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        remainder(x, 0)


def test_remainder_float_zero_divisor_is_nan():
    x = torch.tensor([3.0, -3.0, 0.0])
    y = torch.tensor([0.0, 0.0, 0.0])

    tri = remainder(x, y)

    torch.testing.assert_close(
        tri, torch.remainder(x, y), rtol=0, atol=0, equal_nan=True
    )


def test_remainder_empty():
    x = torch.empty(0, dtype=torch.float32)
    y = torch.empty(0, dtype=torch.float32)

    tri = remainder(x, y)

    assert tri.shape == (0,)
    torch.testing.assert_close(tri, torch.remainder(x, y), rtol=0, atol=0)


@pytest.mark.parametrize("shape_x, shape_y", [((16, 256), (256,)), ((4, 1), (4, 128))])
def test_remainder_broadcast(shape_x, shape_y):
    torch.manual_seed(0)
    x = torch.randn(shape_x, dtype=torch.float32) * 10.0
    y = torch.randn(shape_y, dtype=torch.float32) * 5.0 + 1.0

    tri = remainder(x, y)

    assert tri.shape == torch.remainder(x, y).shape
    torch.testing.assert_close(tri, torch.remainder(x, y), rtol=0, atol=0)


def test_remainder_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.float32).t() * 10.0
    y = torch.randn(32, 64, dtype=torch.float32).t() * 5.0 + 1.0
    assert not x.is_contiguous() and not y.is_contiguous()

    tri = remainder(x, y)

    torch.testing.assert_close(tri, torch.remainder(x, y), rtol=0, atol=0)


def test_remainder_dtype_promotion():
    torch.manual_seed(0)
    x32 = torch.randn(64, dtype=torch.float32) * 10.0
    y64 = torch.randn(64, dtype=torch.float64) * 5.0 + 1.0
    tri = remainder(x32, y64)
    assert tri.dtype == torch.float64
    torch.testing.assert_close(tri, torch.remainder(x32, y64), rtol=0, atol=0)

    xi = torch.randint(-100, 100, (64,), dtype=torch.int32)
    yj = torch.randint(-50, 50, (64,), dtype=torch.int64)
    yj[yj == 0] = 3
    tri = remainder(xi, yj)
    assert tri.dtype == torch.int64
    torch.testing.assert_close(tri, torch.remainder(xi, yj), rtol=0, atol=0)

    # Weak scalars.
    x16 = torch.randn(64, dtype=torch.float16) * 10.0
    assert remainder(x16, 2.5).dtype == torch.float16
    xint, _ = _int_pair(64)
    assert remainder(xint, 2.5).dtype == torch.float32


def test_remainder_both_scalar():
    # torch.remainder rejects two scalars; Python's % carries exactly the
    # divisor-sign semantics remainder needs (upstream uses A % B too).
    assert remainder(7, 2).item() == 1
    assert remainder(-7, 3).item() == 2
    assert remainder(7.5, 2.0).item() == 1.5
    assert remainder(-7.5, -2.0).item() == -1.5


@pytest.mark.parametrize("size", [512, 1024])
def test_remainder__inplace(size):
    x, y = _float_pair(size, torch.float32)
    x_ref = x.clone()

    x_ref.remainder_(y)
    ret = remainder_(x, y)

    assert ret is x
    torch.testing.assert_close(x, x_ref, rtol=0, atol=0)


def test_remainder__inplace_scalar():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32) * 10.0
    x_ref = x.clone()

    x_ref.remainder_(2.5)
    ret = remainder_(x, 2.5)

    assert ret is x
    torch.testing.assert_close(x, x_ref, rtol=0, atol=0)


def test_remainder__inplace_integer():
    # Unlike upstream fmod_, remainder_ serves integer tensors (both
    # upstream and ATen allow it).
    x, y = _int_pair(512)
    x_ref = x.clone()

    x_ref.remainder_(y)
    ret = remainder_(x, y)

    assert ret is x
    torch.testing.assert_close(x, x_ref, rtol=0, atol=0)


def test_remainder__inplace_non_contiguous():
    # Write-through: the in-place result must land in the base storage,
    # including for a genuinely non-contiguous view.
    torch.manual_seed(0)
    base = torch.randn(32, 64, dtype=torch.float32) * 10.0
    x = base[::2, ::2].t()
    y = torch.randn(x.shape, dtype=torch.float32) * 5.0 + 1.0
    assert not x.is_contiguous()

    x_ref = x.clone()
    x_ref.remainder_(y)

    ret = remainder_(x, y)

    assert ret is x
    torch.testing.assert_close(x, x_ref, rtol=0, atol=0)
    torch.testing.assert_close(base[::2, ::2].t(), x_ref, rtol=0, atol=0)


def test_remainder__inplace_empty():
    x = torch.empty(0, dtype=torch.float32)
    y = torch.empty(0, dtype=torch.float32)

    ret = remainder_(x, y)

    assert ret is x
    assert x.numel() == 0


def test_remainder__integer_zero_divisor():
    x, y = _int_pair(16)
    y[3] = 0

    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        remainder_(x, y)


def test_remainder_triton_path(monkeypatch):
    # Smoke: the fp32 main path must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.remainder")

    launches = []
    real_run = mod.remainder_kernel_tt.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.remainder_kernel_tt, "run", spy_run)
    x, y = _float_pair(1024, torch.float32)
    out = mod.remainder(x, y)
    assert len(launches) == 1, "remainder(x, y) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.remainder(x, y), rtol=0, atol=0)
