import pytest
import torch

from .tan import tan, tan_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_tan(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.tan(x)
    tri_out = tan(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_tan_(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x_ref = x.clone()

    x_ref.tan_()
    tan_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
