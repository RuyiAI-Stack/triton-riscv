import pytest
import torch

from .maximum import maximum


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023,), (1024,)])
def test_maximum_tt(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.maximum(x, y)
    tri_out = maximum(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023,), (1024,)])
def test_maximum_ts(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = 5.0

    ref_out = torch.clamp(x, min=y)
    tri_out = maximum(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_maximum_broadcast(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape[1], dtype=torch.float32, device="cpu")

    ref_out = torch.maximum(x, y)
    tri_out = maximum(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
