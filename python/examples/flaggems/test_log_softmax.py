import pytest
import torch

from .log_softmax import (
    log_softmax,
    log_softmax_backward,
    log_softmax_backward_out,
    log_softmax_out,
)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_log_softmax_1d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.nn.functional.log_softmax(x, dim=0)
    tri_out = log_softmax(x, dim=0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape, dim", [((4, 128), 1), ((16, 64), 0), ((8, 32), 1)])
def test_log_softmax_2d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.nn.functional.log_softmax(x, dim=dim)
    tri_out = log_softmax(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


def test_log_softmax_half_to_float():
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float16, device="cpu")
    tri_out = log_softmax(x, dim=1, half_to_float=True)
    assert tri_out.dtype == torch.float32


@pytest.mark.parametrize("shape, dim", [((4, 128), 1), ((16, 64), 0)])
def test_log_softmax_backward(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu", requires_grad=True)
    grad_output = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.nn.functional.log_softmax(x, dim=dim)
    ref_out.backward(grad_output)
    x_grad_ref = x.grad.clone()

    y = log_softmax(x.detach(), dim=dim)
    x_grad_tri = log_softmax_backward(grad_output, y, dim, torch.float32)

    torch.testing.assert_close(x_grad_tri, x_grad_ref, rtol=1e-4, atol=1e-4)


def test_log_softmax_out():
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32, device="cpu")
    out = torch.empty_like(x)
    log_softmax_out(x, dim=1, out=out)
    ref = torch.nn.functional.log_softmax(x, dim=1)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape, dim", [((4, 128), 1), ((16, 64), 0)])
def test_log_softmax_backward_out(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu", requires_grad=True)
    grad_output = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.nn.functional.log_softmax(x, dim=dim)
    ref_out.backward(grad_output)
    x_grad_ref = x.grad.clone()

    y = log_softmax(x.detach(), dim=dim)
    out_grad = torch.empty(shape, dtype=torch.float32, device="cpu")
    x_grad_tri = log_softmax_backward_out(
        grad_output, y, dim, torch.float32, out=out_grad
    )

    torch.testing.assert_close(x_grad_tri, x_grad_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_grad, x_grad_ref, rtol=1e-4, atol=1e-4)
