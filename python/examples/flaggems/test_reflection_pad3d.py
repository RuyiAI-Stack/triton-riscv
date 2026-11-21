import importlib

import pytest
import torch

from .reflection_pad3d import reflection_pad3d, reflection_pad3d_out


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((1, 1, 2, 3, 4), (1, 2, 2, 0, 0, 0)),
        ((2, 2, 3, 3, 4), (1, 1, 1, 1, 1, 1)),
        ((1, 2, 2, 3, 5), (3, 0, 1, 1, 0, 1)),
        ((1, 1, 3, 4, 5), (0, 0, 0, 0, 0, 0)),  # zero padding: copy path
        ((1, 2, 4, 3, 2), (1, 1, 2, 1, 0, 1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_reflection_pad3d_matches_aten(shape, padding, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    out = reflection_pad3d(x, padding)
    ref = torch.ops.aten.reflection_pad3d(x, padding)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == ref.shape


def test_reflection_pad3d_padding_is_aten_ordered():
    # aten order (w0,w1,h0,h1,d0,d1): asymmetric on all three axes.
    x = torch.arange(2 * 2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 2, 3, 4, 5)
    padding = (3, 1, 2, 0, 1, 0)
    out = reflection_pad3d(x, padding)
    ref = torch.ops.aten.reflection_pad3d(x, padding)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad3d_size1_dim_with_zero_pad():
    # aten accepts D == 1 when the depth pads are zero (only pad < dim is
    # checked); upstream rejected every size-1 spatial dim.
    x = torch.arange(1 * 1 * 1 * 3 * 4, dtype=torch.float64).reshape(1, 1, 1, 3, 4)
    out = reflection_pad3d(x, (1, 1, 0, 0, 0, 0))
    ref = torch.ops.aten.reflection_pad3d(x, (1, 1, 0, 0, 0, 0))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (1, 1, 1, 3, 6)


def test_reflection_pad3d_4d_input():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64)  # (C, D, H, W)
    padding = (1, 1, 1, 1, 1, 1)
    out = reflection_pad3d(x, padding)
    ref = torch.ops.aten.reflection_pad3d(x, padding)
    assert out.shape == (2, 5, 6, 7)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad3d_int_dtype():
    x = torch.arange(2 * 2 * 2 * 3 * 3, dtype=torch.int64).reshape(2, 2, 2, 3, 3)
    out = reflection_pad3d(x, (1, 1, 1, 1, 0, 0))
    ref = torch.ops.aten.reflection_pad3d(x, (1, 1, 1, 1, 0, 0))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad3d_out_variant():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 2, 3, 4, dtype=torch.float64)
    ref = torch.ops.aten.reflection_pad3d(x, (1, 1, 1, 0, 0, 0))
    out = torch.empty(ref.shape, dtype=torch.float64)
    ret = reflection_pad3d_out(x, (1, 1, 1, 0, 0, 0), out)
    assert ret is out
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad3d_out_validation():
    x = torch.randn(1, 2, 2, 3, 4, dtype=torch.float64)
    pad = (1, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError, match="out tensor has shape"):
        reflection_pad3d_out(x, pad, torch.empty(1, 2, 2, 3, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="dtype"):
        reflection_pad3d_out(x, pad, torch.empty(1, 2, 2, 4, 6, dtype=torch.float32))


def test_reflection_pad3d_validation_errors():
    x = torch.randn(1, 2, 2, 3, 4)
    with pytest.raises(ValueError, match="sequence of length 6"):
        reflection_pad3d(x, (1, 1, 1, 1))
    with pytest.raises(ValueError, match="must be >= 0"):
        reflection_pad3d(x, (1, -1, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="must be a sequence"):
        reflection_pad3d(x, 2)
    with pytest.raises(ValueError, match="less than the input spatial dimensions"):
        reflection_pad3d(x, (4, 0, 0, 0, 0, 0))  # pad >= dim on width 4
    with pytest.raises(ValueError, match="at least 5 dimensions"):
        reflection_pad3d(torch.randn(2, 3, 4), (1, 1, 1, 1, 1, 1))


def test_reflection_pad3d_triton_path_both_kernels(monkeypatch):
    mod = importlib.import_module("flaggems.reflection_pad3d")
    x = torch.randn(1, 2, 3, 4, 5, dtype=torch.float64)
    for kname, pad in (
        ("reflection_pad3d_kernel", (1, 1, 1, 1, 1, 1)),
        ("copy_tensor_kernel", (0, 0, 0, 0, 0, 0)),
    ):
        launches = []
        kernel = getattr(mod, kname)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        out = reflection_pad3d(x, pad)
        ref = torch.ops.aten.reflection_pad3d(x, pad)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert len(launches) >= 1, f"{kname} must be launched"
