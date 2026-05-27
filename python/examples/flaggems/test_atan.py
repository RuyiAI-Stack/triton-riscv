import pytest
import torch

from .atan import atan, atan_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_atan(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.atan(x)
    tri_out = atan(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_atan_(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x_ref = x.clone()

    x_ref.atan_()
    atan_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
