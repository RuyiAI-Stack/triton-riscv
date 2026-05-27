import pytest
import torch

from .absolute import absolute


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_absolute_float(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.absolute(x)
    tri_out = absolute(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_absolute_complex(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.complex64)

    ref_out = torch.absolute(x)
    tri_out = absolute(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
