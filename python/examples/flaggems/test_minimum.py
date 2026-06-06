import pytest
import torch

from .minimum import minimum


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023,), (1024,)])
def test_minimum_tt(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.minimum(x, y)
    tri_out = minimum(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023,), (1024,)])
def test_minimum_ts(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = -5.0

    ref_out = torch.clamp(x, max=y)
    tri_out = minimum(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_minimum_broadcast(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape[1], dtype=torch.float32, device="cpu")

    ref_out = torch.minimum(x, y)
    tri_out = minimum(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
