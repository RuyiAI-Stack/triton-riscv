import pytest
import torch

from ._prelu_kernel_backward import _prelu_kernel_backward


@pytest.mark.parametrize(
    "shape, weight_shape",
    [
        ((2, 3), (3,)),
        ((2, 3, 171), (171,)),
        ((2, 3, 171), (1,)),
        ((2, 3, 4), (1, 3, 1)),
        ((2, 3, 4), (2, 1, 4)),
        ((2, 1, 4), (1, 3, 1)),
        ((0, 3, 4), (1, 3, 1)),
    ],
)
def test_prelu_kernel_backward_matches_torch(shape, weight_shape):
    torch.manual_seed(0)
    grad = torch.randn(shape, dtype=torch.float32, device="cpu")
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    weight = torch.randn(weight_shape, dtype=torch.float32, device="cpu")

    grad_input, grad_weight = _prelu_kernel_backward(grad, x, weight)
    ref_input, ref_weight = torch.ops.aten._prelu_kernel_backward(grad, x, weight)

    assert grad_input.shape == ref_input.shape
    assert grad_weight.shape == ref_weight.shape
    torch.testing.assert_close(grad_input, ref_input, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad_weight, ref_weight, rtol=1e-4, atol=1e-4)


def test_prelu_kernel_backward_strided_broadcast_weight():
    x = torch.linspace(-1, 1, 24).reshape(2, 3, 4).transpose(0, 2)
    grad = torch.ones_like(x)
    weight = torch.arange(6, dtype=x.dtype)[::2].reshape(1, 3, 1)
    actual = _prelu_kernel_backward(grad, x, weight)
    expected = torch.ops.aten._prelu_kernel_backward(grad, x, weight)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)
