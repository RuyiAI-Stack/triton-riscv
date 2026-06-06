import pytest
import torch

from .celu import celu, celu_


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_celu_forward(shape, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.nn.functional.celu(x, alpha=alpha)
    tri = celu(x, alpha=alpha)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_celu_inplace():
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    torch.nn.functional.celu_(x_ref, alpha=1.0)
    celu_(x, alpha=1.0)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
