import pytest
import torch

from .rad2deg import rad2deg, rad2deg_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_rad2deg(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.rad2deg(x)
    out = rad2deg(x)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_rad2deg_(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()

    x_ref.rad2deg_()
    rad2deg_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-5, atol=1e-5)
