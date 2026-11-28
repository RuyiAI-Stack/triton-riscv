import importlib

import pytest
import torch

from .cast_Long import cast_Long


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_cast_long_matches_aten(dtype):
    if dtype is torch.bool:
        x = (torch.arange(2 * 17) % 3 == 0).reshape(2, 17)
    elif dtype.is_floating_point:
        x = torch.randn(2 * 17, dtype=dtype).reshape(2, 17) * 1e3
    else:
        info = torch.iinfo(dtype)
        lo = max(info.min, -70000)
        hi = min(info.max, 70001)
        x = torch.randint(lo, hi, (2, 17), dtype=dtype)
    out = cast_Long(x)
    ref = torch.ops.aten._cast_Long(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.int64


def test_cast_long_identity_same_dtype():
    x = torch.randint(-(2**40), 2**40, (3, 4), dtype=torch.int64)
    assert cast_Long(x) is x


def test_cast_long_negative_float_truncation():
    # fptosi truncates toward zero: -3.9 -> -3, 3.9 -> 3.
    x = torch.tensor([-3.9, 3.9, -0.9, 0.9, -1e12, 1e12], dtype=torch.float64)
    torch.testing.assert_close(
        cast_Long(x), torch.ops.aten._cast_Long(x), rtol=0, atol=0
    )


def test_cast_long_fp32_through_widen():
    # float32 goes through the same float32->int64 fptosi as aten.
    x = torch.tensor([2.5e9, -2.5e9, 1.5, -1.5], dtype=torch.float32)
    torch.testing.assert_close(
        cast_Long(x), torch.ops.aten._cast_Long(x), rtol=0, atol=0
    )


def test_cast_long_int_float_int_roundtrip():
    xi = torch.randint(-(2**52), 2**52, (5, 9), dtype=torch.int64)
    back = cast_Long(xi.to(torch.float64))
    torch.testing.assert_close(back, xi, rtol=0, atol=0)


def test_cast_long_strided_input_both_kernels():
    # A transpose of a contiguous tensor is still non-overlapping-and-dense:
    # it takes the flat kernel (a linear storage-to-storage copy is exact for
    # any dense permutation). Sliced / expanded views are non-dense and take
    # the strided decode kernel.
    xt = torch.randn(4, 6).t()
    assert not xt.is_contiguous()
    assert torch.ops.aten.is_non_overlapping_and_dense(xt)
    torch.testing.assert_close(
        cast_Long(xt), torch.ops.aten._cast_Long(xt), rtol=0, atol=0
    )
    xs = torch.randn(4, 8)[::2, ::2]  # non-dense strided view
    assert not torch.ops.aten.is_non_overlapping_and_dense(xs)
    torch.testing.assert_close(
        cast_Long(xs), torch.ops.aten._cast_Long(xs), rtol=0, atol=0
    )
    y = torch.randn(9)[::3]
    torch.testing.assert_close(
        cast_Long(y), torch.ops.aten._cast_Long(y), rtol=0, atol=0
    )


def test_cast_long_boundary_shapes():
    x0 = torch.empty(0, 5, dtype=torch.float32)
    assert cast_Long(x0).shape == (0, 5)
    x1 = torch.tensor([42.7], dtype=torch.float64)
    torch.testing.assert_close(
        cast_Long(x1), torch.ops.aten._cast_Long(x1), rtol=0, atol=0
    )
    scalar = torch.tensor(-5.5, dtype=torch.float32)
    torch.testing.assert_close(
        cast_Long(scalar), torch.ops.aten._cast_Long(scalar), rtol=0, atol=0
    )


def test_cast_long_expanded_input():
    base = torch.tensor([[1.25, -2.75]], dtype=torch.float32)
    x = base.expand(4, 2)
    torch.testing.assert_close(
        cast_Long(x), torch.ops.aten._cast_Long(x), rtol=0, atol=0
    )


def test_cast_long_type_errors():
    with pytest.raises(TypeError, match="invalid combination of arguments"):
        cast_Long([1.0, 2.0])
    with pytest.raises(TypeError, match="non_blocking must be a bool"):
        cast_Long(torch.tensor([1.0]), non_blocking="yes")


def test_cast_long_triton_path_both_kernels(monkeypatch):
    mod = importlib.import_module("flaggems.cast_Long")
    x = torch.randn(4, 2050, dtype=torch.float32) * 100
    for kname in ("_cast_long_flat_kernel", "_cast_long_strided_kernel"):
        launches = []
        kernel = getattr(mod, kname)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        inp = x[::2] if "strided" in kname else x
        out = cast_Long(inp)
        ref = torch.ops.aten._cast_Long(inp)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert len(launches) >= 1, f"{kname} must be launched"
