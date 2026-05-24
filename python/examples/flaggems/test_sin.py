import pytest
import torch

from .sin import sin, sin_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_sin(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.sin(x)
    tri_out = sin(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_sin_(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x_ref = x.clone()

    x_ref.sin_()
    sin_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
