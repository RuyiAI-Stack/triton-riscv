import pytest
import torch

from .arcsinh import arcsinh, arcsinh_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arcsinh(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.arcsinh(x)
    tri_out = arcsinh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arcsinh_out(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.arcsinh(x)
    out = torch.empty_like(x)
    arcsinh_out(x, out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
