import pytest
import torch

from .ceil import ceil, ceil_, ceil_out


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ceil(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu") * 10
    ref = torch.ceil(x)
    tri = ceil(x)
    torch.testing.assert_close(tri, ref)


def test_ceil_inplace():
    x = torch.tensor([1.1, 2.9, -0.5, -1.3], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    x_ref.ceil_()
    ceil_(x)
    torch.testing.assert_close(x, x_ref)


def test_ceil_out():
    x = torch.tensor([1.1, 2.9, -0.5], dtype=torch.float32, device="cpu")
    out = torch.empty_like(x)
    ceil_out(x, out=out)
    torch.testing.assert_close(out, torch.ceil(x))
