import importlib

import pytest
import torch

from .cast_Float import cast_Float


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float64,
    ],
)
def test_cast_float_matches_aten(dtype):
    if dtype is torch.bool:
        x = (torch.arange(2 * 17) % 3 == 0).reshape(2, 17)
    elif dtype.is_floating_point:
        x = torch.randn(2 * 17, dtype=dtype).reshape(2, 17)
    else:
        info = torch.iinfo(dtype)
        lo = max(info.min, -70000)
        hi = min(info.max, 70001)
        x = torch.randint(lo, hi, (2, 17), dtype=dtype)
    out = cast_Float(x)
    ref = torch.ops.aten._cast_Float(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.float32


def test_cast_float_identity_same_dtype():
    x = torch.randn(3, 4, dtype=torch.float32)
    assert cast_Float(x) is x


def test_cast_float_large_int64_rounds_to_nearest():
    # int64 -> float32 rounds to the nearest representable value, like aten.
    x = torch.tensor([2**31 - 1, -(2**31), 16777217, 16777216], dtype=torch.int64)
    torch.testing.assert_close(
        cast_Float(x), torch.ops.aten._cast_Float(x), rtol=0, atol=0
    )


def test_cast_float_fp64_downcast_precision():
    x = torch.tensor([0.1, 1e-40, 3.141592653589793], dtype=torch.float64)
    torch.testing.assert_close(
        cast_Float(x), torch.ops.aten._cast_Float(x), rtol=0, atol=0
    )


def test_cast_float_strided_input_both_kernels():
    # A transpose of a contiguous tensor is still non-overlapping-and-dense:
    # it takes the flat kernel (a linear storage-to-storage copy is exact for
    # any dense permutation). Sliced / expanded views are non-dense and take
    # the strided decode kernel.
    xt = torch.randn(4, 6).t()
    assert not xt.is_contiguous()
    assert torch.ops.aten.is_non_overlapping_and_dense(xt)
    torch.testing.assert_close(
        cast_Float(xt), torch.ops.aten._cast_Float(xt), rtol=0, atol=0
    )
    xs = torch.randn(4, 8)[::2, ::2]  # non-dense strided view
    assert not torch.ops.aten.is_non_overlapping_and_dense(xs)
    torch.testing.assert_close(
        cast_Float(xs), torch.ops.aten._cast_Float(xs), rtol=0, atol=0
    )
    y = torch.randn(9)[::3]
    torch.testing.assert_close(
        cast_Float(y), torch.ops.aten._cast_Float(y), rtol=0, atol=0
    )


def test_cast_float_boundary_shapes():
    x0 = torch.empty(3, 0, dtype=torch.float64)
    assert cast_Float(x0).shape == (3, 0)
    x1 = torch.tensor([2**-149], dtype=torch.float64)  # subnormal fp32
    torch.testing.assert_close(
        cast_Float(x1), torch.ops.aten._cast_Float(x1), rtol=0, atol=0
    )


def test_cast_float_requires_grad_fallback_and_values():
    x = torch.randn(4, dtype=torch.float64, requires_grad=True)
    out = cast_Float(x)
    torch.testing.assert_close(out, torch.ops.aten._cast_Float(x), rtol=0, atol=0)
    with torch.no_grad():
        out2 = cast_Float(x)
        torch.testing.assert_close(out2, x.float(), rtol=0, atol=0)


def test_cast_float_complex_falls_back_to_aten():
    x = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128)
    out = cast_Float(x)
    torch.testing.assert_close(out, torch.ops.aten._cast_Float(x), rtol=0, atol=0)
    assert out.dtype == torch.float32


def test_cast_float_triton_path_both_kernels(monkeypatch):
    mod = importlib.import_module("flaggems.cast_Float")
    x = torch.randn(4, 2050, dtype=torch.float64)
    for kname in ("_cast_float_kernel", "_cast_float_strided"):
        launches = []
        kernel = getattr(mod, kname)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        inp = x[::2] if kname.endswith("strided") else x
        out = cast_Float(inp)
        ref = torch.ops.aten._cast_Float(inp)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert len(launches) >= 1, f"{kname} must be launched"
