import pytest
import torch

from .relu import relu, relu_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_relu(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.relu(x)
    tri = relu(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_relu_inplace():
    x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    torch.nn.functional.relu(x_ref, inplace=True)
    relu_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
