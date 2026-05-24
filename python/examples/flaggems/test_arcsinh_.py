import pytest
import torch

from .arcsinh_ import arcsinh_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arcsinh_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x_ref = x.clone()

    x_ref.arcsinh_()
    arcsinh_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
