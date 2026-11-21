import importlib

import pytest
import torch

from .cast_Short import cast_Short


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_cast_short_matches_aten(dtype):
    if dtype is torch.bool:
        x = (torch.arange(2 * 17) % 3 == 0).reshape(2, 17)
    elif dtype.is_floating_point:
        x = torch.randn(2 * 17, dtype=dtype).reshape(2, 17) * 100
    else:
        info = torch.iinfo(dtype)
        lo = max(info.min, -70000)
        hi = min(info.max, 70001)
        x = torch.randint(lo, hi, (2, 17), dtype=dtype)
    out = cast_Short(x)
    ref = torch.ops.aten._cast_Short(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.int16


def test_cast_short_identity_same_dtype():
    x = torch.randint(-32768, 32768, (3, 4), dtype=torch.int16)
    assert cast_Short(x) is x


def test_cast_short_int16_boundary_wraparound():
    # x.to(int32).to(int16) wraps: 65535 -> -1, 32768 -> -32768, fp16 65504 -> -32.
    xi = torch.tensor([65535, 65536, 32768, -32769], dtype=torch.int64)
    torch.testing.assert_close(
        cast_Short(xi), torch.ops.aten._cast_Short(xi), rtol=0, atol=0
    )
    xh = torch.tensor([65504.0, -65504.0, 700.0], dtype=torch.float16)
    torch.testing.assert_close(
        cast_Short(xh), torch.ops.aten._cast_Short(xh), rtol=0, atol=0
    )


def test_cast_short_negative_float_truncation():
    x = torch.tensor([-3.9, 3.9, -700.2, 700.9], dtype=torch.float64)
    torch.testing.assert_close(
        cast_Short(x), torch.ops.aten._cast_Short(x), rtol=0, atol=0
    )


def test_cast_short_complex_keeps_real_part():
    x = torch.tensor([1 + 2j, 3 - 4j, -5.5 + 0j], dtype=torch.complex64)
    out = cast_Short(x)
    torch.testing.assert_close(out, torch.ops.aten._cast_Short(x), rtol=0, atol=0)
    assert out.dtype == torch.int16


def test_cast_short_strided_input_both_kernels():
    # A transpose of a contiguous tensor is still non-overlapping-and-dense:
    # it takes the flat kernel (a linear storage-to-storage copy is exact for
    # any dense permutation). Sliced / expanded views are non-dense and take
    # the strided decode kernel.
    xt = torch.randn(4, 6).t()
    assert not xt.is_contiguous()
    assert torch.ops.aten.is_non_overlapping_and_dense(xt)
    torch.testing.assert_close(
        cast_Short(xt), torch.ops.aten._cast_Short(xt), rtol=0, atol=0
    )
    xs = torch.randn(4, 8)[::2, ::2]  # non-dense strided view
    assert not torch.ops.aten.is_non_overlapping_and_dense(xs)
    torch.testing.assert_close(
        cast_Short(xs), torch.ops.aten._cast_Short(xs), rtol=0, atol=0
    )
    y = torch.randn(9)[::3]
    torch.testing.assert_close(
        cast_Short(y), torch.ops.aten._cast_Short(y), rtol=0, atol=0
    )


def test_cast_short_boundary_shapes():
    x0 = torch.empty(2, 0, dtype=torch.float32)
    assert cast_Short(x0).shape == (2, 0)
    x1 = torch.tensor([40000.5], dtype=torch.float64)
    torch.testing.assert_close(
        cast_Short(x1), torch.ops.aten._cast_Short(x1), rtol=0, atol=0
    )
    scalar = torch.tensor(2.9, dtype=torch.float32)
    torch.testing.assert_close(
        cast_Short(scalar), torch.ops.aten._cast_Short(scalar), rtol=0, atol=0
    )


def test_cast_short_type_errors():
    with pytest.raises(TypeError, match="must be a Tensor"):
        cast_Short(3.5)
    with pytest.raises(TypeError, match="must be a bool"):
        cast_Short(torch.tensor([1.0]), non_blocking=None)


def test_cast_short_triton_path_both_kernels(monkeypatch):
    mod = importlib.import_module("flaggems.cast_Short")
    x = torch.randn(4, 2050, dtype=torch.float32) * 100
    for kname in ("_cast_short_flat_kernel", "_cast_short_strided_kernel"):
        launches = []
        kernel = getattr(mod, kname)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        inp = x[::2] if "strided" in kname else x
        out = cast_Short(inp)
        ref = torch.ops.aten._cast_Short(inp)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert len(launches) >= 1, f"{kname} must be launched"
