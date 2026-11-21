import importlib

import pytest
import torch

from .cast_Char import cast_Char


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
def test_cast_char_matches_aten(dtype):
    if dtype is torch.bool:
        x = (torch.arange(2 * 17) % 3 == 0).reshape(2, 17)
    elif dtype.is_floating_point:
        x = torch.randn(2 * 17, dtype=dtype).reshape(2, 17) * 100
    else:
        info = torch.iinfo(dtype)
        lo = max(info.min, -70000)
        hi = min(info.max, 70001)
        x = torch.randint(lo, hi, (2, 17), dtype=dtype)
    out = cast_Char(x)
    ref = torch.ops.aten._cast_Char(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.int8


def test_cast_char_identity_same_dtype():
    x = torch.randint(-128, 128, (3, 4), dtype=torch.int8)
    assert cast_Char(x) is x


def test_cast_char_wraparound_semantics():
    # Floats narrow through int32 to signed int8.
    x = torch.tensor([-258.75, 300.5, -1.5, 127.5, -0.5, 65535.9], dtype=torch.float64)
    torch.testing.assert_close(
        cast_Char(x), torch.ops.aten._cast_Char(x), rtol=0, atol=0
    )
    xi = torch.tensor([-(2**40), 2**40 + 300, -1, 128], dtype=torch.int64)
    torch.testing.assert_close(
        cast_Char(xi), torch.ops.aten._cast_Char(xi), rtol=0, atol=0
    )


def test_cast_char_int_float_int_roundtrip():
    xi = torch.randint(-128, 128, (5, 9), dtype=torch.int8)
    back = cast_Char(xi.to(torch.float64))
    torch.testing.assert_close(back, xi, rtol=0, atol=0)


def test_cast_char_strided_input_both_kernels():
    # A transpose of a contiguous tensor is still non-overlapping-and-dense:
    # it takes the flat kernel (a linear storage-to-storage copy is exact for
    # any dense permutation). Sliced / expanded views are non-dense and take
    # the strided decode kernel.
    xt = torch.randn(4, 6).t()
    assert not xt.is_contiguous()
    assert torch.ops.aten.is_non_overlapping_and_dense(xt)
    torch.testing.assert_close(
        cast_Char(xt), torch.ops.aten._cast_Char(xt), rtol=0, atol=0
    )
    xs = torch.randn(4, 8)[::2, ::2]  # non-dense strided view
    assert not torch.ops.aten.is_non_overlapping_and_dense(xs)
    torch.testing.assert_close(
        cast_Char(xs), torch.ops.aten._cast_Char(xs), rtol=0, atol=0
    )
    y = torch.randn(9)[::3]
    torch.testing.assert_close(
        cast_Char(y), torch.ops.aten._cast_Char(y), rtol=0, atol=0
    )


def test_cast_char_boundary_shapes():
    x0 = torch.empty(0, dtype=torch.float32)
    assert cast_Char(x0).shape == (0,)
    x1 = torch.tensor([300.9], dtype=torch.float32)
    torch.testing.assert_close(
        cast_Char(x1), torch.ops.aten._cast_Char(x1), rtol=0, atol=0
    )


def test_cast_char_complex_falls_back_to_aten():
    x = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128)
    out = cast_Char(x)
    torch.testing.assert_close(out, torch.ops.aten._cast_Char(x), rtol=0, atol=0)
    assert out.dtype == torch.int8


def test_cast_char_triton_path_both_kernels(monkeypatch):
    mod = importlib.import_module("flaggems.cast_Char")
    x = torch.randn(4, 2050, dtype=torch.float32) * 100
    for kname in ("_cast_char_kernel", "_cast_char_strided"):
        launches = []
        kernel = getattr(mod, kname)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        inp = x[::2] if kname.endswith("strided") else x
        out = cast_Char(inp)
        ref = torch.ops.aten._cast_Char(inp)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert len(launches) >= 1, f"{kname} must be launched"
