import pytest
import torch

from .sigmoid import sigmoid, sigmoid_, sigmoid_backward


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_sigmoid(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    ref = torch.sigmoid(x)
    tri = sigmoid(x)

    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_sigmoid_inplace(dtype):
    x = torch.tensor([0.0, 1.0, -1.0], dtype=dtype, device="cpu")
    x_ref = x.clone()
    torch.sigmoid(x_ref, out=x_ref)
    sigmoid_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sigmoid_backward(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu", requires_grad=True)

    out = torch.sigmoid(x)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    ref_grad = x.grad.clone()

    x.grad.zero_()
    out = sigmoid(x)
    tri_grad = sigmoid_backward(grad_output, out)

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-3, atol=1e-3)
