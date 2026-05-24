import pytest
import torch

from .tanh import tanh, tanh_, tanh_backward


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_tanh(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.tanh(x)
    tri = tanh(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_tanh_inplace():
    x = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone().tanh_()
    tanh_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_tanh_backward(shape):
    torch.manual_seed(0)
    x = torch.randn(
        shape, dtype=torch.float32, device="cpu", requires_grad=True
    )
    grad_output = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.tanh(x)
    ref.backward(grad_output)
    ref_grad = x.grad.clone()

    y = tanh(x.detach())
    tri_grad = tanh_backward(grad_output, y)

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-4, atol=1e-4)
