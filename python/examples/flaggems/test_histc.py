import pytest
import torch

from .histc import histc


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_histc_default_bins(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.histc(x)
    tri_out = histc(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_histc_50_bins(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.histc(x, bins=50)
    tri_out = histc(x, bins=50)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_histc_200_bins(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.histc(x, bins=200)
    tri_out = histc(x, bins=200)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_histc_custom_range(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.histc(x, bins=100, min=-1.0, max=1.0)
    tri_out = histc(x, bins=100, min=-1.0, max=1.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_histc_constant():
    torch.manual_seed(0)
    x = torch.ones(1024, device="cpu", dtype=torch.float32) * 3.0

    ref_out = torch.histc(x, bins=10, min=0.0, max=5.0)
    tri_out = histc(x, bins=10, min=0.0, max=5.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
