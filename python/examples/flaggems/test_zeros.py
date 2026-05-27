import pytest
import torch

from .zeros import zero_, zeros


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_zeros(shape):
    tri_out = zeros(shape, dtype=torch.float32, device="cpu")
    ref_out = torch.zeros(shape, dtype=torch.float32, device="cpu")
    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_zero_(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x_ref = x.clone()

    x_ref.zero_()
    zero_(x)

    torch.testing.assert_close(x, x_ref, rtol=0, atol=0)
