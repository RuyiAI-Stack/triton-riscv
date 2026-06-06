import pytest
import torch

from .cos import cos, cos_


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
def test_cos(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.cos(x)
    tri = cos(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_cos_inplace():
    x = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    torch.cos(x_ref, out=x_ref)
    cos_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
