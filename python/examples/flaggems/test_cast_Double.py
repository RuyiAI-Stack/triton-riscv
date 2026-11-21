import importlib

import pytest
import torch

from .cast_Double import cast_Double


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
        torch.float32,
    ],
)
def test_cast_double_matches_aten(dtype):
    if dtype is torch.bool:
        x = (torch.arange(2 * 17) % 3 == 0).reshape(2, 17)
    elif dtype.is_floating_point:
        x = torch.randn(2 * 17, dtype=dtype).reshape(2, 17)
    else:
        info = torch.iinfo(dtype)
        lo = max(info.min, -70000)
        hi = min(info.max, 70001)
        x = torch.randint(lo, hi, (2, 17), dtype=dtype)
    out = cast_Double(x)
    ref = torch.ops.aten._cast_Double(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.float64


def test_cast_double_identity_same_dtype():
    x = torch.randn(3, 4, dtype=torch.float64)
    assert cast_Double(x) is x


def test_cast_double_large_int64_exact():
    # fp64 has a 53-bit mantissa; large int64 values are still represented.
    x = torch.tensor([2**53 - 1, -(2**53), 2**40 + 7], dtype=torch.int64)
    torch.testing.assert_close(
        cast_Double(x), torch.ops.aten._cast_Double(x), rtol=0, atol=0
    )


def test_cast_double_int_float_int_roundtrip():
    xi = torch.randint(-(2**31), 2**31, (5, 9), dtype=torch.int64)
    back = cast_Double(cast_Double(xi)).to(torch.int64)
    torch.testing.assert_close(back, xi, rtol=0, atol=0)


def test_cast_double_strided_input_both_kernels():
    # A transpose of a contiguous tensor is still non-overlapping-and-dense:
    # it takes the flat kernel (a linear storage-to-storage copy is exact for
    # any dense permutation). Sliced / expanded views are non-dense and take
    # the strided decode kernel.
    xt = torch.randn(4, 6).t()
    assert not xt.is_contiguous()
    assert torch.ops.aten.is_non_overlapping_and_dense(xt)
    torch.testing.assert_close(
        cast_Double(xt), torch.ops.aten._cast_Double(xt), rtol=0, atol=0
    )
    xs = torch.randn(4, 8)[::2, ::2]  # non-dense strided view
    assert not torch.ops.aten.is_non_overlapping_and_dense(xs)
    torch.testing.assert_close(
        cast_Double(xs), torch.ops.aten._cast_Double(xs), rtol=0, atol=0
    )
    y = torch.randn(9)[::3]
    torch.testing.assert_close(
        cast_Double(y), torch.ops.aten._cast_Double(y), rtol=0, atol=0
    )


def test_cast_double_boundary_shapes():
    x0 = torch.empty(0, 3, dtype=torch.float32)
    assert cast_Double(x0).shape == (0, 3)
    x1 = torch.tensor([1e-300], dtype=torch.float32)
    torch.testing.assert_close(
        cast_Double(x1), torch.ops.aten._cast_Double(x1), rtol=0, atol=0
    )


def test_cast_double_requires_grad_fallback_and_values():
    x = torch.randn(4, dtype=torch.float32, requires_grad=True)
    assert torch.is_grad_enabled()
    out = cast_Double(x)  # grad-enabled requires_grad input -> aten fallback
    torch.testing.assert_close(out, torch.ops.aten._cast_Double(x), rtol=0, atol=0)
    with torch.no_grad():
        out2 = cast_Double(x)  # no_grad -> Triton path taken
        torch.testing.assert_close(out2, x.double(), rtol=0, atol=0)


def test_cast_double_complex_falls_back_to_aten():
    x = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64)
    out = cast_Double(x)
    torch.testing.assert_close(out, torch.ops.aten._cast_Double(x), rtol=0, atol=0)
    assert out.dtype == torch.float64


def test_cast_double_triton_path_both_kernels(monkeypatch):
    mod = importlib.import_module("flaggems.cast_Double")
    x = torch.randn(4, 2050, dtype=torch.float32)
    for kname in ("_cast_double_kernel", "_cast_double_strided"):
        launches = []
        kernel = getattr(mod, kname)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        inp = x[::2] if kname.endswith("strided") else x
        out = cast_Double(inp)
        ref = torch.ops.aten._cast_Double(inp)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert len(launches) >= 1, f"{kname} must be launched"
