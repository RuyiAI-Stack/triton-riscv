import importlib

import pytest
import torch
import torch.nn.functional as F

from .reflection_pad3d_backward import reflection_pad3d_backward


def _run(x, padding, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    d_out = x.shape[-3] + padding[4] + padding[5]
    h_out = x.shape[-2] + padding[2] + padding[3]
    w_out = x.shape[-1] + padding[0] + padding[1]
    grad = torch.randn(*x.shape[:-3], d_out, h_out, w_out, dtype=dtype)
    out = reflection_pad3d_backward(grad, x, padding)
    ref = torch.ops.aten.reflection_pad3d_backward(grad, x, padding)
    return out, ref


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((1, 1, 4, 3, 5), (1, 1, 1, 0, 1, 0)),
        ((2, 2, 3, 3, 4), (1, 1, 1, 1, 1, 1)),
        ((1, 2, 2, 4, 3), (2, 1, 3, 0, 0, 1)),
        ((1, 1, 4, 4, 4), (2, 2, 2, 2, 2, 2)),  # corner voxels collect 27 terms
        ((1, 2, 3, 4, 5), (0, 0, 0, 0, 0, 0)),  # zero padding: clone path
    ],
)
def test_reflection_pad3d_backward_matches_aten(shape, padding):
    x = torch.randn(shape, dtype=torch.float64)
    out, ref = _run(x, padding)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad3d_backward_padding_is_aten_ordered():
    # The wrapper must take (w0, w1, h0, h1, d0, d1) like aten; upstream's
    # depth-first unpack would swap the roles of the pads here.
    x = torch.randn(1, 1, 4, 3, 5, dtype=torch.float64)
    padding = (3, 1, 2, 0, 1, 0)  # asymmetric on all three axes
    out, ref = _run(x, padding)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad3d_backward_fp32_fp16_exact():
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(1, 2, 4, 4, 4, dtype=dtype)
        torch.manual_seed(1)
        grad = torch.randn(1, 2, 6, 6, 6, dtype=dtype)
        out = reflection_pad3d_backward(grad, x, (1, 1, 1, 1, 1, 1))
        ref = torch.ops.aten.reflection_pad3d_backward(grad, x, (1, 1, 1, 1, 1, 1))
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad3d_backward_autograd_crosscheck():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(1, 2, 4, 6, 7, dtype=torch.float64)
    y = F.pad(x, (1, 1, 1, 1, 1, 0), mode="reflect")
    y.backward(grad)
    out = reflection_pad3d_backward(grad, x.detach(), (1, 1, 1, 1, 1, 0))
    torch.testing.assert_close(out, x.grad, rtol=0, atol=0)


def test_reflection_pad3d_backward_4d_input():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64)  # (C, D, H, W)
    grad = torch.randn(2, 5, 6, 7, dtype=torch.float64)
    out = reflection_pad3d_backward(grad, x, (1, 1, 1, 1, 1, 1))
    ref = torch.ops.aten.reflection_pad3d_backward(grad, x, (1, 1, 1, 1, 1, 1))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == x.shape


def test_reflection_pad3d_backward_int_padding_broadcast():
    # Upstream (and this port) broadcast a scalar padding to all six sides;
    # aten requires a 6-list, so the oracle expands the same padding.
    x = torch.randn(1, 1, 4, 4, 4, dtype=torch.float64)
    torch.manual_seed(0)
    grad = torch.randn(1, 1, 6, 6, 6, dtype=torch.float64)
    out = reflection_pad3d_backward(grad, x, 1)
    ref = torch.ops.aten.reflection_pad3d_backward(grad, x, (1,) * 6)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad3d_backward_validation_errors():
    x = torch.randn(1, 2, 3, 4, 5)
    with pytest.raises(ValueError, match="input must be a 5D tensor"):
        reflection_pad3d_backward(
            torch.randn(2, 4, 6, 7, 8), torch.randn(2, 3), (1, 1, 1, 1, 1, 1)
        )
    grad_bad = torch.randn(1, 2, 4, 4, 4)  # wrong shape for padding (1,1,1,1,1,1)
    with pytest.raises(ValueError, match="grad_output has shape"):
        reflection_pad3d_backward(grad_bad, x, (1, 1, 1, 1, 1, 1))


def test_reflection_pad3d_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.reflection_pad3d_backward")
    launches = []
    kernel = mod.reflection_pad3d_backward_kernel
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x = torch.randn(1, 2, 4, 4, 4, dtype=torch.float64)
    out, ref = _run(x, (1, 1, 1, 1, 1, 1))
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
