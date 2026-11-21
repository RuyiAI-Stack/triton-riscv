import importlib

import pytest
import torch

from .cast_Byte import cast_Byte


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
        torch.float64,
    ],
)
def test_cast_byte_matches_aten(dtype):
    if dtype is torch.bool:
        x = (torch.arange(2 * 17) % 3 == 0).reshape(2, 17)
    elif dtype.is_floating_point:
        x = torch.randn(2 * 17, dtype=dtype).reshape(2, 17) * 100
    else:
        info = torch.iinfo(dtype)
        lo = max(info.min, -70000)
        hi = min(info.max, 70001)
        x = torch.randint(lo, hi, (2, 17), dtype=dtype)
    out = cast_Byte(x)
    ref = torch.ops.aten._cast_Byte(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.uint8


def test_cast_byte_identity_same_dtype():
    x = torch.randint(0, 255, (3, 4), dtype=torch.uint8)
    assert cast_Byte(x) is x


def test_cast_byte_wraparound_semantics():
    # PyTorch narrows floats through int64 then truncates to uint8.
    x = torch.tensor([-258.75, 300.5, -1.5, 255.5, -0.5, 65535.9], dtype=torch.float64)
    torch.testing.assert_close(
        cast_Byte(x), torch.ops.aten._cast_Byte(x), rtol=0, atol=0
    )
    xi = torch.tensor([-(2**40), 2**40 + 5, -1, 256], dtype=torch.int64)
    torch.testing.assert_close(
        cast_Byte(xi), torch.ops.aten._cast_Byte(xi), rtol=0, atol=0
    )


def test_cast_byte_int_float_int_roundtrip():
    xi = torch.randint(0, 256, (5, 9), dtype=torch.uint8)
    mid = xi.to(torch.float64)
    assert (mid == xi.double()).all()
    back = cast_Byte(mid)
    torch.testing.assert_close(back, xi, rtol=0, atol=0)


def test_cast_byte_strided_input_both_kernels():
    # A transpose of a contiguous tensor is still non-overlapping-and-dense:
    # it takes the flat kernel (a linear storage-to-storage copy is exact for
    # any dense permutation). Sliced / expanded views are non-dense and take
    # the strided decode kernel.
    xt = torch.randn(4, 6).t()
    assert not xt.is_contiguous()
    assert torch.ops.aten.is_non_overlapping_and_dense(xt)
    torch.testing.assert_close(
        cast_Byte(xt), torch.ops.aten._cast_Byte(xt), rtol=0, atol=0
    )
    xs = torch.randn(4, 8)[::2, ::2]  # non-dense strided view
    assert not torch.ops.aten.is_non_overlapping_and_dense(xs)
    torch.testing.assert_close(
        cast_Byte(xs), torch.ops.aten._cast_Byte(xs), rtol=0, atol=0
    )
    y = torch.randn(9)[::3]
    torch.testing.assert_close(
        cast_Byte(y), torch.ops.aten._cast_Byte(y), rtol=0, atol=0
    )


def test_cast_byte_boundary_shapes():
    x0 = torch.empty(0, 4, dtype=torch.float32)
    assert cast_Byte(x0).shape == (0, 4)
    x1 = torch.tensor([3.9], dtype=torch.float32)
    torch.testing.assert_close(
        cast_Byte(x1), torch.ops.aten._cast_Byte(x1), rtol=0, atol=0
    )
    scalar = torch.tensor(200.4, dtype=torch.float64)
    torch.testing.assert_close(
        cast_Byte(scalar), torch.ops.aten._cast_Byte(scalar), rtol=0, atol=0
    )


def test_cast_byte_expanded_input():
    base = torch.tensor([[1.5, -2.5]], dtype=torch.float32)
    x = base.expand(3, 2)  # stride-0 broadcast, non-dense
    torch.testing.assert_close(
        cast_Byte(x), torch.ops.aten._cast_Byte(x), rtol=0, atol=0
    )


def test_cast_byte_complex_falls_back_to_aten():
    # Complex is not on the Triton path; the aten fallback discards the
    # imaginary part (same warning as aten's own cast).
    x = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64)
    out = cast_Byte(x)
    torch.testing.assert_close(out, torch.ops.aten._cast_Byte(x), rtol=0, atol=0)
    assert out.dtype == torch.uint8


def test_cast_byte_triton_path_both_kernels(monkeypatch):
    mod = importlib.import_module("flaggems.cast_Byte")
    x = torch.randn(4, 2050, dtype=torch.float32) * 1000
    for kname in ("_cast_byte_kernel", "_cast_byte_strided"):
        launches = []
        kernel = getattr(mod, kname)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        if kname.endswith("strided"):
            inp = x[::2]
        else:
            inp = x
        out = cast_Byte(inp)
        ref = torch.ops.aten._cast_Byte(inp)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert len(launches) >= 1, f"{kname} must be launched"
