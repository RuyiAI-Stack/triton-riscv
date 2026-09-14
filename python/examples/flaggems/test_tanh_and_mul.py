import pytest
import torch

from .tanh_and_mul import tanh_and_mul, tanh_and_mul_backward


@pytest.mark.parametrize(
    "x_shape,y_shape",
    [
        ((512,), (512,)),
        ((1023,), (1023,)),
        ((1024,), (1024,)),
        ((4, 1), (1, 8)),
        ((1, 17), (3, 1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tanh_and_mul_forward(x_shape, y_shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(x_shape, dtype=dtype, device="cpu")
    y = torch.randn(y_shape, dtype=dtype, device="cpu")

    expected = torch.tanh(x) * y
    actual = tanh_and_mul(x, y)

    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    "x_shape,y_shape",
    [
        ((512,), (512,)),
        ((1023,), (1023,)),
        ((1024,), (1024,)),
        ((4, 1), (1, 8)),
        ((1, 17), (3, 1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tanh_and_mul_backward(x_shape, y_shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(x_shape, dtype=dtype, device="cpu", requires_grad=True)
    y = torch.randn(y_shape, dtype=dtype, device="cpu", requires_grad=True)

    expected = torch.tanh(x) * y
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    expected_dx = x.grad.clone()
    expected_dy = y.grad.clone()

    actual_dx, actual_dy = tanh_and_mul_backward(
        grad_output, x.detach(), y.detach()
    )

    torch.testing.assert_close(actual_dx, expected_dx, rtol=0.01, atol=0.01)
    torch.testing.assert_close(actual_dy, expected_dy, rtol=0.01, atol=0.01)
