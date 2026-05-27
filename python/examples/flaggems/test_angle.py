import pytest
import torch

from .angle import angle


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_angle_real(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.angle(x)
    tri_out = angle(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_angle_complex(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.complex64)

    ref_out = torch.angle(x)
    tri_out = angle(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
