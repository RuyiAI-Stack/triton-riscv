import pytest
import torch

from .std import std


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_std_1d(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")

    ref_out = torch.std(x)
    tri_out = std(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_std_1d_keepdim(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")

    ref_out = torch.std(x, keepdim=True)
    tri_out = std(x, keepdim=True)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_std_1d_correction(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")

    ref_out = torch.std(x, correction=0.0)
    tri_out = std(x, correction=0.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(64, 8), (32, 32), (128, 8)])
def test_std_2d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.std(x, dim=1)
    tri_out = std(x, dim=1)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(64, 8), (32, 32), (128, 8)])
def test_std_2d_keepdim(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.std(x, dim=1, keepdim=True)
    tri_out = std(x, dim=1, keepdim=True)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
