import pytest
import torch

from .zero import zero, zero_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_zero(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x_ref = x.clone()

    ref_out = x_ref.zero_()
    tri_out = zero(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 64), (4, 32, 64)])
def test_zero_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)
    x_ref = x.clone()

    ref_out = x_ref.zero_()
    out = torch.empty(shape, device="cpu", dtype=torch.float32)
    tri_out = zero_out(out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 64), (4, 32, 64)])
def test_zero_2d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)
    x_ref = x.clone()

    ref_out = x_ref.zero_()
    tri_out = zero(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
