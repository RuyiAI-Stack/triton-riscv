import pytest
import torch

from .copysign import copysign, copysign_out


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
def test_copysign_tt(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.sign(torch.randn(shape, device="cpu"))
    ref = torch.copysign(x, y)
    tri = copysign(x, y)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_copysign_ts():
    x = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32, device="cpu")
    ref = torch.copysign(x, -1.0)
    tri = copysign(x, -1.0)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
def test_copysign_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.sign(torch.randn(shape, device="cpu"))
    ref = torch.copysign(x, y)
    out = torch.empty(shape, dtype=torch.float32, device="cpu")
    tri = copysign_out(x, y, out=out)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_copysign_out_scalar():
    x = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32, device="cpu")
    ref = torch.copysign(x, -1.0)
    out = torch.empty(4, dtype=torch.float32, device="cpu")
    tri = copysign_out(x, -1.0, out=out)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
