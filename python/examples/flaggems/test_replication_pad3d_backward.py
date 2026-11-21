import importlib

import pytest
import torch
import torch.nn.functional as F

from .replication_pad3d_backward import replication_pad3d_backward


def _make(shape, padding, dtype, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype)
    d_out = shape[-3] + padding[4] + padding[5]
    h_out = shape[-2] + padding[2] + padding[3]
    w_out = shape[-1] + padding[0] + padding[1]
    grad = torch.randn(*shape[:-3], d_out, h_out, w_out, dtype=dtype)
    return x, grad


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((1, 1, 4, 3, 5), (1, 1, 1, 0, 1, 0)),
        ((2, 2, 3, 3, 4), (1, 1, 1, 1, 1, 1)),
        ((1, 2, 2, 4, 3), (2, 1, 3, 0, 0, 1)),
        ((1, 1, 4, 4, 4), (2, 2, 2, 2, 2, 2)),  # corner voxels collect 27 terms
        ((1, 2, 3, 4, 5), (0, 0, 0, 0, 0, 0)),  # zero padding: copy path
        ((2, 3, 4, 5), (1, 1, 1, 1, 1, 1)),  # unbatched (C, D, H, W)
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_replication_pad3d_backward_matches_aten(shape, padding, dtype):
    x, grad = _make(shape, padding, dtype)
    out = replication_pad3d_backward(grad, x, padding)
    ref = torch.ops.aten.replication_pad3d_backward(grad, x, padding)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_replication_pad3d_backward_low_precision_exact(dtype):
    x, grad = _make((1, 2, 4, 4, 4), (1, 1, 1, 1, 1, 1), dtype)
    out = replication_pad3d_backward(grad, x, (1, 1, 1, 1, 1, 1))
    ref = torch.ops.aten.replication_pad3d_backward(grad, x, (1, 1, 1, 1, 1, 1))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_replication_pad3d_backward_padding_is_aten_ordered():
    x, grad = _make((1, 1, 4, 3, 5), (3, 1, 2, 0, 1, 0), torch.float64)
    out = replication_pad3d_backward(grad, x, (3, 1, 2, 0, 1, 0))
    ref = torch.ops.aten.replication_pad3d_backward(grad, x, (3, 1, 2, 0, 1, 0))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_replication_pad3d_backward_autograd_crosscheck():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(1, 2, 4, 6, 7, dtype=torch.float64)
    y = F.pad(x, (1, 1, 1, 1, 1, 0), mode="replicate")
    y.backward(grad)
    out = replication_pad3d_backward(grad, x.detach(), (1, 1, 1, 1, 1, 0))
    torch.testing.assert_close(out, x.grad, rtol=0, atol=0)


def test_replication_pad3d_backward_corner_sum_order():
    # Corner (0,0,0) with pads (2,2,2,2,2,2) sums the 3x3x3 block of
    # grad_output at (d,h,w) in {0,1,2}^3, in aten's ascending order.
    torch.manual_seed(3)
    x = torch.zeros(1, 1, 4, 4, 4, dtype=torch.float64)
    grad = torch.randn(1, 1, 8, 8, 8, dtype=torch.float64)
    out = replication_pad3d_backward(grad, x, (2, 2, 2, 2, 2, 2))
    ref = torch.ops.aten.replication_pad3d_backward(grad, x, (2, 2, 2, 2, 2, 2))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    expected = sum(
        grad[0, 0, dd, hh, ww]
        for dd in (0, 1, 2)
        for hh in (0, 1, 2)
        for ww in (0, 1, 2)
    )
    assert out[0, 0, 0, 0, 0] == expected


def test_replication_pad3d_backward_zero_padding_copy_path(monkeypatch):
    mod = importlib.import_module("flaggems.replication_pad3d_backward")
    launches = []
    kernel = mod._copy_rows_kernel
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x, grad = _make((1, 2, 3, 4, 5), (0, 0, 0, 0, 0, 0), torch.float64)
    out = replication_pad3d_backward(grad, x, (0, 0, 0, 0, 0, 0))
    torch.testing.assert_close(out, grad, rtol=0, atol=0)
    assert len(launches) >= 1


def test_replication_pad3d_backward_validation_errors():
    x = torch.randn(1, 2, 3, 4, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="sequence of length 6"):
        replication_pad3d_backward(torch.randn(1, 2, 5, 6, 7), x, (1, 1, 1, 1))
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        replication_pad3d_backward(torch.randn(6), torch.randn(6), (1, 1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="replication_pad3d_backward supports float16"):
        replication_pad3d_backward(
            torch.randn(1, 2, 5, 6, 7, dtype=torch.complex128),
            x.to(torch.complex128),
            (1, 1, 1, 1, 1, 1),
        )
    with pytest.raises(ValueError, match="same device and dtype"):
        replication_pad3d_backward(
            torch.randn(1, 2, 5, 6, 7, dtype=torch.float32), x, (1, 1, 1, 1, 1, 1)
        )
    with pytest.raises(ValueError, match="does not match expected"):
        replication_pad3d_backward(
            torch.randn(1, 2, 4, 6, 7, dtype=torch.float64), x, (1, 1, 1, 1, 1, 1)
        )
    with pytest.raises(ValueError, match="matching leading dimensions"):
        replication_pad3d_backward(
            torch.randn(2, 2, 5, 6, 7, dtype=torch.float64), x, (1, 1, 1, 1, 1, 1)
        )


def test_replication_pad3d_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.replication_pad3d_backward")
    launches = []
    kernel = mod._replication_pad3d_backward_kernel
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x, grad = _make((1, 2, 4, 4, 4), (1, 1, 1, 1, 1, 1), torch.float64)
    out = replication_pad3d_backward(grad, x, (1, 1, 1, 1, 1, 1))
    ref = torch.ops.aten.replication_pad3d_backward(grad, x, (1, 1, 1, 1, 1, 1))
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
