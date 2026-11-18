import math

import pytest
import torch

from .xlogy import (
    xlogy,
    xlogy_out,
    xlogy_scalar_tensor,
    xlogy_scalar_tensor_out,
    xlogy_tensor_scalar,
    xlogy_tensor_scalar_out,
)

# Tolerances (source: the kernel computes x * log(y) in fp32 for every input
# dtype -- upstream casts both operands to fp32 -- while torch CPU computes
# in the tensor dtype, fp32 opmath for fp16):
#   fp32: a couple of log ulp -> 1e-4, the repo test_arctan2.py convention.
#   fp16: both sides round the same fp32 value to fp16 -> 1e-3 (fp16 eps).
#   fp64: kernel is fp32 -> ~1e-7 relative vs torch's fp64 reference -> 1e-6
#         (test_arctan2.py precedent for an always-fp32 kernel).


def _tolerance(dtype):
    if dtype == torch.float16:
        return 1e-3, 1e-3
    if dtype == torch.float64:
        return 1e-6, 1e-6
    return 1e-4, 1e-4


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_xlogy_tt(dtype, size):
    torch.manual_seed(0)
    # Random y includes negatives: both sides produce NaN there.
    x = torch.randn(size, dtype=dtype) * 10.0
    y = torch.randn(size, dtype=dtype) * 5.0

    ref = torch.xlogy(x, y)
    tri = xlogy(x, y)

    rtol, atol = _tolerance(dtype)
    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_xlogy_tensor_scalar(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 10.0
    y = 2.5

    ref = torch.xlogy(x, y)
    tri = xlogy_tensor_scalar(x, y)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_xlogy_scalar_tensor(size):
    torch.manual_seed(0)
    x = 0.5
    y = torch.randn(size, dtype=torch.float32) * 5.0

    ref = torch.xlogy(x, y)
    tri = xlogy_scalar_tensor(x, y)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize(
    "shape_x, shape_y", [((16, 256), (256,)), ((4, 1), (4, 128))]
)
def test_xlogy_broadcast(shape_x, shape_y):
    torch.manual_seed(0)
    x = torch.randn(shape_x, dtype=torch.float32) * 10.0
    y = torch.randn(shape_y, dtype=torch.float32) * 5.0

    ref = torch.xlogy(x, y)
    tri = xlogy(x, y)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_xlogy_special_values():
    # ATen precedence, probed on torch CPU fp32:
    #   1. y is NaN -> NaN (even when x == 0: xlogy(0, nan) == nan);
    #   2. x == +-0 -> +0.0 for every other y (including -inf and 0);
    #   3. otherwise x * log(y) in plain IEEE arithmetic.
    # Every row below lands on an exact value (0 / +-inf / NaN), so an
    # exact comparison is meaningful.
    pos0, neg0 = 0.0, -0.0
    inf, ninf = float("inf"), float("-inf")
    nan = float("nan")
    x = torch.tensor(
        [pos0, pos0, neg0, pos0, pos0, pos0, 3.0, 3.0, inf, inf, ninf, 2.0, 1.0, nan, 2.0],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [nan, ninf, 5.0, pos0, 1.0, inf, pos0, ninf, 0.5, inf, inf, nan, 1.0, 1.0, -1.0],
        dtype=torch.float32,
    )

    ref = torch.xlogy(x, y)
    tri = xlogy(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    # Pin the zero-row contract exactly: NaN wins over x == 0, and the
    # zero row comes back as +0.0 (never -0.0 or NaN) for finite/inf y.
    assert math.isnan(tri[0].item())  # xlogy(0, nan) == nan
    assert tri[1].item() == 0.0  # xlogy(0, -inf) == 0
    assert math.copysign(1.0, tri[2].item()) > 0  # xlogy(-0, 5) == +0
    assert math.copysign(1.0, tri[4].item()) > 0  # xlogy(0, 1) == +0
    assert tri[6].item() == float("-inf")  # xlogy(3, 0)
    assert tri[12].item() == 0.0  # xlogy(1, 1) == 0
    assert tri[14].item() != tri[14].item()  # xlogy(2, -1) == nan (y < 0)


def test_xlogy_y_equals_one():
    # xlogy(*, 1) reduces to 0 exactly.
    x = torch.randn(1023, dtype=torch.float32) * 10.0
    y = torch.ones(1023, dtype=torch.float32)

    tri = xlogy(x, y)

    torch.testing.assert_close(tri, torch.zeros_like(x), rtol=0, atol=0)


def test_xlogy_fp16_specials():
    x = torch.tensor([0.0, 2.0, -1.0], dtype=torch.float16)
    y = torch.tensor(
        [float("nan"), 0.0, 2.0], dtype=torch.float16
    )

    ref = torch.xlogy(x, y)
    tri = xlogy(x, y)

    assert tri.dtype == torch.float16
    assert math.isnan(tri[0].item())
    assert tri[1].item() == float("-inf")
    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3, equal_nan=True)


def test_xlogy_empty():
    x = torch.empty(0, dtype=torch.float32)
    y = torch.empty(0, dtype=torch.float32)

    ref = torch.xlogy(x, y)
    tri = xlogy(x, y)

    assert tri.shape == (0,)
    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_xlogy_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.float32).t() * 10.0
    y = torch.randn(32, 64, dtype=torch.float32).t() * 5.0
    assert not x.is_contiguous() and not y.is_contiguous()

    ref = torch.xlogy(x, y)
    tri = xlogy(x, y)

    assert tri.shape == x.shape
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_xlogy_int_promotes_to_float():
    # torch type promotion: xlogy(int, int) -> float32 (upstream declares
    # the INT_TO_FLOAT promotion).
    torch.manual_seed(0)
    x = torch.randint(0, 9, (512,), dtype=torch.int64)
    y = torch.randint(1, 9, (512,), dtype=torch.int64)

    ref = torch.xlogy(x, y)
    tri = xlogy(x, y)

    assert tri.dtype == torch.float32
    assert ref.dtype == torch.float32
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_xlogy_mixed_int_widths():
    x = torch.arange(1, 6, dtype=torch.int32) * 2
    y = torch.arange(1, 6, dtype=torch.int64)

    ref = torch.xlogy(x, y)
    tri = xlogy(x, y)

    assert tri.dtype == torch.float32
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_xlogy_weak_scalar_promotion():
    # torch keeps a floating tensor's dtype next to a Python scalar and
    # promotes an integer tensor to fp32 (INT_TO_FLOAT).
    x16 = torch.randn(64, dtype=torch.float16) + 2.0
    assert xlogy_tensor_scalar(x16, 2.5).dtype == torch.float16

    xi = torch.arange(1, 8, dtype=torch.int64)
    assert xlogy_tensor_scalar(xi, 2.5).dtype == torch.float32
    assert xlogy_tensor_scalar(xi, 3).dtype == torch.float32

    y64 = torch.randn(64, dtype=torch.float64) + 2.0
    assert xlogy_scalar_tensor(2.5, y64).dtype == torch.float64
    yi = torch.arange(1, 8, dtype=torch.int64)
    assert xlogy_scalar_tensor(2, yi).dtype == torch.float32


def test_xlogy_out():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32) * 10.0
    y = torch.randn(1024, dtype=torch.float32) * 5.0
    out = torch.empty(1024, dtype=torch.float32)

    ref = torch.xlogy(x, y)
    ret = xlogy_out(x, y, out)

    assert ret is out
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_xlogy_out_scalar_overloads():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32) * 10.0
    y = torch.randn(512, dtype=torch.float32) * 5.0

    ref_ts = torch.xlogy(x, 2.5)
    out_ts = torch.empty(512, dtype=torch.float32)
    ret_ts = xlogy_tensor_scalar_out(x, 2.5, out_ts)
    assert ret_ts is out_ts
    torch.testing.assert_close(out_ts, ref_ts, rtol=1e-4, atol=1e-4)

    ref_st = torch.xlogy(0.5, y)
    out_st = torch.empty(512, dtype=torch.float32)
    ret_st = xlogy_scalar_tensor_out(0.5, y, out_st)
    assert ret_st is out_st
    torch.testing.assert_close(out_st, ref_st, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_xlogy_triton_path(monkeypatch):
    # Smoke: the fp32 main path must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.xlogy")

    launches = []
    real_run = mod.xlogy_kernel_tt.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.xlogy_kernel_tt, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 10.0
    y = torch.randn(1024, dtype=torch.float32) * 5.0
    out = mod.xlogy(x, y)
    assert len(launches) == 1, "xlogy(x, y) must go through the Triton kernel"
    torch.testing.assert_close(out, torch.xlogy(x, y), rtol=1e-4, atol=1e-4, equal_nan=True)
