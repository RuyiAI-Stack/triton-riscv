import pytest
import torch

from .reciprocal import reciprocal, reciprocal_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_reciprocal(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu") + 0.5
    ref = torch.reciprocal(x)
    tri = reciprocal(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_reciprocal_inplace():
    x = torch.tensor([0.5, 2.0, 4.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone().reciprocal_()
    reciprocal_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
