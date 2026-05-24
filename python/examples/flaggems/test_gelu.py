import pytest
import torch

from .gelu import gelu, gelu_, gelu_backward


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_gelu_none(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.gelu(x, approximate="none")
    tri = gelu(x, approximate="none")

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_gelu_tanh(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.gelu(x, approximate="tanh")
    tri = gelu(x, approximate="tanh")

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_gelu_inplace():
    for approx in ["none", "tanh"]:
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device="cpu")
        x_ref = x.clone()
        gelu_(x, approximate=approx)
        x_ref = torch.nn.functional.gelu(x_ref, approximate=approx)
        torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(64, 128), (128, 128)])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_backward(shape, approximate):
    torch.manual_seed(0)
    x = torch.randn(
        shape, dtype=torch.float32, device="cpu", requires_grad=True
    )
    grad_output = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.gelu(x, approximate=approximate)
    ref.backward(grad_output)
    x_grad_ref = x.grad.clone()

    x_grad_tri = gelu_backward(
        grad_output, x.detach(), approximate=approximate
    )

    torch.testing.assert_close(x_grad_tri, x_grad_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64]
)
def test_gelu_dtype(dtype):
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=dtype, device="cpu")

    ref = torch.nn.functional.gelu(x, approximate="none")
    tri = gelu(x, approximate="none")

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
