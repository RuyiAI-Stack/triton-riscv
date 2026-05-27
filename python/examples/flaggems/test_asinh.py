import pytest
import torch

from .asinh import asinh, asinh_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_asinh(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.asinh(x)
    tri_out = asinh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_asinh_out(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    out = torch.empty_like(x)

    ref_out = torch.asinh(x)
    asinh_out(x, out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
