import math

import pytest
import torch

from .arctan2 import arctan2, arctan2_, arctan2_out

# Tolerances (source: both sides compute atan2 in fp32 -- the Triton kernel
# casts to fp32 and calls libdevice.atan2, mirroring upstream arctan2_kernel,
# and torch's CPU fp32 atan2 is fp32-accurate):
#   fp32: rtol=atol=1e-4, matching the existing repo test_atan2.py convention
#         (actual agreement is a couple of fp32 ulp, ~2.4e-7 at |result| <= pi).
#   fp16: rtol=atol=1e-3, fp16 eps ~ 9.8e-4; both sides round fp32 -> fp16.
#   fp64: rtol=atol=1e-6, the kernel computes in fp32 (upstream casts to
#         fp32), so ~1e-7 relative accuracy vs torch's fp64 reference.


def _tolerance(dtype):
    if dtype == torch.float16:
        return 1e-3, 1e-3
    if dtype == torch.float64:
        return 1e-6, 1e-6
    return 1e-4, 1e-4


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_arctan2_tt(dtype, size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=dtype, device="cpu")
    y = torch.randn(size, dtype=dtype, device="cpu")

    ref = torch.arctan2(x, y)
    tri = arctan2(x, y)

    rtol, atol = _tolerance(dtype)
    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arctan2_ts(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")
    y = 0.5

    ref = torch.arctan2(x, torch.tensor(y))
    tri = arctan2(x, y)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arctan2_st(size):
    torch.manual_seed(0)
    x = 0.5
    y = torch.randn(size, dtype=torch.float32, device="cpu")

    ref = torch.arctan2(torch.tensor(x), y)
    tri = arctan2(x, y)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape_A, shape_B", [((16, 256), (256,)), ((4, 1), (4, 128))])
def test_arctan2_broadcast(shape_A, shape_B):
    torch.manual_seed(0)
    x = torch.randn(shape_A, dtype=torch.float32, device="cpu")
    y = torch.randn(shape_B, dtype=torch.float32, device="cpu")

    ref = torch.arctan2(x, y)
    tri = arctan2(x, y)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_arctan2_empty():
    x = torch.empty(0, dtype=torch.float32, device="cpu")
    y = torch.empty(0, dtype=torch.float32, device="cpu")

    ref = torch.arctan2(x, y)
    tri = arctan2(x, y)

    assert tri.shape == (0,)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_arctan2_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.float32, device="cpu").t()
    y = torch.randn(32, 64, dtype=torch.float32, device="cpu").t()
    assert not x.is_contiguous() and not y.is_contiguous()

    ref = torch.arctan2(x, y)
    tri = arctan2(x, y)

    assert tri.shape == x.shape
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_arctan2_int_promotes_to_float():
    # torch type promotion: atan2(int, int) -> float32 (upstream declares the
    # INT_TO_FLOAT promotion for the kernel).
    torch.manual_seed(0)
    x = torch.randint(-9, 9, (512,), dtype=torch.int64, device="cpu")
    y = torch.randint(1, 9, (512,), dtype=torch.int64, device="cpu")

    ref = torch.arctan2(x, y)
    tri = arctan2(x, y)

    assert tri.dtype == torch.float32
    assert ref.dtype == torch.float32
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_arctan2_out():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
    y = torch.randn(1024, dtype=torch.float32, device="cpu")
    out = torch.empty(1024, dtype=torch.float32, device="cpu")

    ref = torch.arctan2(x, y)
    ret = arctan2_out(x, y, out)

    assert ret is out
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_arctan2_zero_and_inf_boundaries():
    # IEEE 754 / C99 annex F specials, which ATen follows: the results are
    # exact multiples of pi/4 (or NaN), so an exact comparison is meaningful.
    pos0, neg0 = 0.0, -0.0
    inf, ninf = float("inf"), float("-inf")
    nan = float("nan")
    x = torch.tensor(
        [
            pos0,
            pos0,
            neg0,
            neg0,
            pos0,
            neg0,
            inf,
            inf,
            ninf,
            inf,
            ninf,
            1.0,
            -1.0,
            nan,
            1.0,
        ],
        dtype=torch.float32,
        device="cpu",
    )
    y = torch.tensor(
        [
            pos0,
            neg0,
            pos0,
            neg0,
            inf,
            inf,
            pos0,
            neg0,
            pos0,
            ninf,
            ninf,
            inf,
            inf,
            1.0,
            nan,
        ],
        dtype=torch.float32,
        device="cpu",
    )

    ref = torch.arctan2(x, y)
    tri = arctan2(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    # Pin the sign-of-zero quadrant rules exactly.
    assert math.copysign(1.0, tri[0].item()) > 0  # atan2(+0, +0) = +0
    assert math.copysign(1.0, tri[2].item()) < 0  # atan2(-0, +0) = -0
    assert math.copysign(1.0, tri[1].item()) > 0  # atan2(+0, -0) = +pi
    assert math.copysign(1.0, tri[3].item()) < 0  # atan2(-0, -0) = -pi
    assert tri[1].item() == float(torch.tensor(math.pi, dtype=torch.float32))


@pytest.mark.parametrize("size", [512, 1024])
def test_arctan2_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")
    y = torch.randn(size, dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    y_ref = y.clone()
    x_ref.arctan2_(y_ref)

    ret = arctan2_(x, y)

    assert ret is x
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


def test_arctan2_inplace_scalar():
    # torch's Tensor.arctan2_ itself only accepts a Tensor other; the scalar
    # path here is checked against the equivalent out-of-place reference.
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
    ref = torch.arctan2(x.clone(), torch.tensor(0.5))

    ret = arctan2_(x, 0.5)

    assert ret is x
    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)


def test_arctan2_inplace_non_contiguous():
    torch.manual_seed(0)
    base = torch.randn(32, 64, dtype=torch.float32, device="cpu")
    x = base[::2, ::2].t()  # genuinely non-contiguous view
    y = torch.randn(x.shape, dtype=torch.float32, device="cpu")
    assert not x.is_contiguous()

    x_ref = x.clone()
    x_ref.arctan2_(y)

    ret = arctan2_(x, y)

    assert ret is x
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


def test_arctan2_inplace_empty():
    x = torch.empty(0, dtype=torch.float32, device="cpu")
    y = torch.empty(0, dtype=torch.float32, device="cpu")

    ret = arctan2_(x, y)

    assert ret is x
    assert x.numel() == 0


def test_arctan2_inplace_rejects_integer():
    # ATen rejects in-place arctan2 on integer output (the Float result cannot
    # be cast back); upstream mirrors the same RuntimeError.
    x = torch.randint(-9, 9, (16,), dtype=torch.int64, device="cpu")
    y = torch.randint(1, 9, (16,), dtype=torch.int64, device="cpu")

    with pytest.raises(RuntimeError, match="can't be cast"):
        arctan2_(x, y)
