import pytest
import torch

from .sigmoid_and_mul import sigmoid_and_mul, sigmoid_and_mul_backward


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
    ],
)
def test_sigmoid_and_mul_forward(shape, dtype, rtol, atol):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    y = torch.randn(shape, dtype=dtype, device="cpu")

    ref = torch.sigmoid(x) * y
    tri = sigmoid_and_mul(x, y)

    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_sigmoid_and_mul_backward(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu", requires_grad=True)
    y = torch.randn(shape, dtype=torch.float32, device="cpu", requires_grad=True)

    ref = torch.sigmoid(x) * y
    grad_output = torch.randn_like(ref)
    ref.backward(grad_output)
    ref_dx = x.grad.clone()
    ref_dy = y.grad.clone()

    tri_dx, tri_dy = sigmoid_and_mul_backward(grad_output, x.detach(), y.detach())

    torch.testing.assert_close(tri_dx, ref_dx, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_dy, ref_dy, rtol=1e-4, atol=1e-4)
