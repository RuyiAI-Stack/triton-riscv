import importlib

import pytest
import torch
import torch.nn.functional as F

from .replication_pad2d_backward import (
    replication_pad2d_backward,
    replication_pad2d_backward_grad_input,
)


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((2, 5, 6), (1, 1, 1, 1)),
        ((1, 2, 5, 6), (2, 2, 1, 1)),
        ((2, 3, 6, 5), (3, 1, 2, 2)),
        ((1, 2, 4, 5), (2, 2, 2, 2)),  # corner cells collect 3x3 = 9 terms
        ((2, 3, 4, 5), (0, 0, 0, 0)),  # zero padding: copy path
        ((2, 5, 6), (1, 2, 0, 3)),
        ((2, 4, 4), (2, 1, 1, 2)),  # unbatched (C, H, W)
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_replication_pad2d_backward_matches_aten(shape, padding, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    oh = shape[-2] + padding[2] + padding[3]
    ow = shape[-1] + padding[0] + padding[1]
    grad = torch.randn(*shape[:-2], oh, ow, dtype=dtype)
    out = replication_pad2d_backward(grad, x, padding)
    ref = torch.ops.aten.replication_pad2d_backward(grad, x, padding)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_replication_pad2d_backward_fp16_exact():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 5, dtype=torch.float16)
    grad = torch.randn(1, 2, 7, 7, dtype=torch.float16)
    out = replication_pad2d_backward(grad, x, (1, 1, 2, 1))
    ref = torch.ops.aten.replication_pad2d_backward(grad, x, (1, 1, 2, 1))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_replication_pad2d_backward_autograd_crosscheck():
    torch.manual_seed(0)
    x = torch.randn(2, 2, 5, 6, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(2, 2, 10, 8, dtype=torch.float64)
    y = F.pad(x, (1, 1, 3, 2), mode="replicate")
    y.backward(grad)
    out = replication_pad2d_backward(grad, x.detach(), (1, 1, 3, 2))
    torch.testing.assert_close(out, x.grad, rtol=0, atol=0)


def test_replication_pad2d_backward_corner_sum_order():
    # (0, 0) sums a 3x3 block of grad_output rows 0..1 x cols 0..1 plus the
    # center row/col; with pads (2,2,2,2) it is the full 3x3 top-left block.
    torch.manual_seed(3)
    x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    grad = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    out = replication_pad2d_backward(grad, x, (2, 2, 2, 2))
    ref = torch.ops.aten.replication_pad2d_backward(grad, x, (2, 2, 2, 2))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    expected = sum(grad[0, 0, r, c] for r in (0, 1, 2) for c in (0, 1, 2))
    assert out[0, 0, 0, 0] == expected  # aten's ascending (row-major) order


def test_replication_pad2d_backward_grad_input_variant():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 5, 6, dtype=torch.float64)
    grad = torch.randn(1, 2, 7, 9, dtype=torch.float64)  # OW = 6 + 1 + 2
    grad_input = torch.empty_like(x)
    ret = replication_pad2d_backward_grad_input(
        grad, x, (1, 2, 1, 1), grad_input=grad_input
    )
    assert ret is grad_input
    ref = torch.ops.aten.replication_pad2d_backward(grad, x, (1, 2, 1, 1))
    torch.testing.assert_close(grad_input, ref, rtol=0, atol=0)


def test_replication_pad2d_backward_int_padding_broadcast():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 5, dtype=torch.float64)
    grad = torch.randn(1, 2, 6, 7, dtype=torch.float64)
    out = replication_pad2d_backward(grad, x, 1)
    ref = torch.ops.aten.replication_pad2d_backward(grad, x, (1, 1, 1, 1))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_replication_pad2d_backward_validation_errors():
    x = torch.randn(1, 2, 4, 5)
    with pytest.raises(ValueError, match="sequence of 4 integers"):
        replication_pad2d_backward(torch.randn(1, 2, 6, 7), x, (1, 1, 1))
    with pytest.raises(ValueError, match="must be non-negative"):
        replication_pad2d_backward(torch.randn(1, 2, 6, 3), x, (-1, 1, 1, 1))
    with pytest.raises(TypeError, match="Unexpected padding type"):
        replication_pad2d_backward(torch.randn(1, 2, 6, 7), x, "abcd")
    with pytest.raises(ValueError, match="3D .C, H, W. or 4D"):
        replication_pad2d_backward(torch.randn(6), torch.randn(6), (1, 1, 1, 1))
    with pytest.raises(ValueError, match="does not match"):
        replication_pad2d_backward(torch.randn(1, 2, 6, 6), x, (1, 1, 1, 1))


def test_replication_pad2d_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.replication_pad2d_backward")
    launches = []
    kernel = mod._replication_pad2d_backward_kernel
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(2, 2, 6, 7, dtype=torch.float64)
    grad = torch.randn(2, 2, 9, 9, dtype=torch.float64)
    out = replication_pad2d_backward(grad, x, (1, 1, 2, 1))
    ref = torch.ops.aten.replication_pad2d_backward(grad, x, (1, 1, 2, 1))
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
