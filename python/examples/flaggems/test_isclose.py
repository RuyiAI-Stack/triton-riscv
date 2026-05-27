import pytest
import torch

from .isclose import allclose, isclose


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_isclose(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.isclose(x, y)
    tri_out = isclose(x, y)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("size", [512, 1024])
def test_allclose(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = x + 1e-6

    ref_out = torch.allclose(x, y, rtol=1e-4, atol=1e-4)
    tri_out = allclose(x, y, rtol=1e-4, atol=1e-4)
    assert tri_out == ref_out

    # Test exact match
    assert allclose(x, x, rtol=0, atol=0)

    # Test far apart
    z = x + 10.0
    assert not allclose(x, z, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_isclose_close_values(size):
    torch.manual_seed(0)
    x = torch.ones(size, device="cpu", dtype=torch.float32)
    y = x + 1e-6

    ref_out = torch.isclose(x, y)
    tri_out = isclose(x, y)

    torch.testing.assert_close(tri_out, ref_out)
