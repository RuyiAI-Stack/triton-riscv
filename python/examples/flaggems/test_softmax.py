import pytest
import torch

from .softmax import (
    softmax,
    softmax_backward,
    softmax_backward_out,
    softmax_out,
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_softmax(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")

    ref_out = torch.softmax(x, dim=-1)
    tri_out = softmax(x, dim=-1)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_softmax_2d(size):
    torch.manual_seed(0)
    x = torch.randn(size, size, dtype=torch.float32, device="cpu")

    ref_out = torch.softmax(x, dim=-1)
    tri_out = softmax(x, dim=-1)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_softmax_backward(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu", requires_grad=True)
    grad_output = torch.randn(size, dtype=torch.float32, device="cpu")

    output = torch.softmax(x, dim=-1)
    ref_grad = torch.autograd.grad(output.sum(), x, create_graph=True)[0]

    # Compute expected gradient manually
    scale = (output * grad_output).sum()
    ref_grad_manual = output * (grad_output - scale)

    tri_grad = softmax_backward(
        grad_output, output, dim=-1, input_dtype=x.dtype
    )

    torch.testing.assert_close(tri_grad, ref_grad_manual, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1024])
def test_softmax_out(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")
    out = torch.empty(size, dtype=torch.float32, device="cpu")

    ref_out = torch.softmax(x, dim=-1)
    tri_out = softmax_out(x, dim=-1, out=out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1024])
def test_softmax_backward_out(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu", requires_grad=True)
    grad_output = torch.randn(size, dtype=torch.float32, device="cpu")

    output = torch.softmax(x, dim=-1)

    # Compute expected gradient
    scale = (output * grad_output).sum()
    ref_grad_manual = output * (grad_output - scale)

    grad_input = torch.empty(size, dtype=torch.float32, device="cpu")
    tri_grad = softmax_backward_out(
        grad_output, output, dim=-1, input_dtype=x.dtype, grad_input=grad_input
    )

    torch.testing.assert_close(tri_grad, ref_grad_manual, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        grad_input, ref_grad_manual, rtol=1e-4, atol=1e-4
    )
