import importlib

import pytest
import torch

from ._pad_circular import _pad_circular


@pytest.mark.parametrize(
    "shape, pad",
    [
        ((2, 4), (1, 2)),
        ((3, 5), (3, 0)),
        ((2, 4), (0, 0)),
        ((2, 4), (4, 0)),  # pad == input size: wraps exactly once
        ((2, 3, 4), (1, 2)),
        ((2, 3, 4), (0, 3)),
        ((2, 3, 4), (1, 2, 0, 1)),
        ((1, 2, 3, 4), (1, 2, 0, 1, 1, 0)),
        ((2, 3, 4, 5), (2, 1, 1, 2, 0, 1)),
    ],
)
def test_pad_circular_matches_aten(shape, pad):
    x = torch.arange(torch.Size(shape).numel(), dtype=torch.float64).reshape(shape)
    out = _pad_circular(x, pad)
    ref = torch.ops.aten._pad_circular(x, pad)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == ref.shape


@pytest.mark.parametrize(
    "pad, width",
    [
        ((-1, 2), 8),  # crop-then-wrap inside the cropped tensor (aten order)
        ((2, -1), 8),
        ((-3, 4), 8),
        ((-2, -2), 8),
        ((-4, 4), 8),
        ((-5, 0), 8),
        ((0, -3), 8),
        ((-1, -1), 8),
        ((3, -5), 8),
    ],
)
def test_pad_circular_negative_padding_matches_aten(pad, width):
    x = torch.arange(2 * width, dtype=torch.float64).reshape(2, width)
    out = _pad_circular(x, pad)
    ref = torch.ops.aten._pad_circular(x, pad)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_pad_circular_empty_output_dim():
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    out = _pad_circular(x, (-9, 1))  # new_size == 0 -> empty (aten agrees)
    assert out.shape == (1, 0)
    out2 = _pad_circular(x, (-2, -2))
    assert out2.shape == (1, 4)
    torch.testing.assert_close(
        out2, torch.ops.aten._pad_circular(x, (-2, -2)), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.int32])
def test_pad_circular_dtypes(dtype):
    x = torch.arange(2 * 6, dtype=dtype).reshape(2, 6)
    torch.testing.assert_close(
        _pad_circular(x, (2, 1)),
        torch.ops.aten._pad_circular(x, (2, 1)),
        rtol=0,
        atol=0,
    )


def test_pad_circular_wrap_error_matches_aten_message():
    x = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
    with pytest.raises(ValueError, match="wrapping around more than once"):
        _pad_circular(x, (5, 0))
    with pytest.raises(ValueError, match="wrapping around more than once"):
        _pad_circular(x, (1, 5))
    with pytest.raises(ValueError, match="wrapping around more than once"):
        # negative crop shrinks the dim below the positive pad
        _pad_circular(x, (-3, 4))


def test_pad_circular_validation_errors():
    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match="padding length must be even"):
        _pad_circular(x, (1, 2, 3))
    with pytest.raises(ValueError, match="last 1, 2 or 3 dimensions"):
        _pad_circular(x, ())
    with pytest.raises(ValueError, match="at least one non-padded"):
        _pad_circular(torch.randn(4), (1, 1))
    with pytest.raises(
        ValueError, match="negative padding removed the whole dimension"
    ):
        _pad_circular(torch.arange(8.0).reshape(1, 8), (-5, -5))


def test_pad_circular_empty_batch_rows():
    x = torch.empty(0, 4, dtype=torch.float32)
    out = _pad_circular(x, (1, 1))
    assert out.shape == (0, 6)


def test_pad_circular_triton_path_all_three_kernels(monkeypatch):
    mod = importlib.import_module("flaggems._pad_circular")
    cases = {
        "_pad_circular_1d_kernel": ((2, 3, 5), (1, 2)),
        "_pad_circular_2d_kernel": ((2, 3, 5), (1, 2, 0, 1)),
        "_pad_circular_3d_kernel": ((1, 2, 3, 4), (1, 1, 1, 1, 1, 1)),
    }
    for kname, (shape, pad) in cases.items():
        launches = []
        kernel = getattr(mod, kname)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        x = torch.arange(torch.Size(shape).numel(), dtype=torch.float64).reshape(shape)
        out = _pad_circular(x, pad)
        ref = torch.ops.aten._pad_circular(x, pad)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert len(launches) >= 1, f"{kname} must be launched"
