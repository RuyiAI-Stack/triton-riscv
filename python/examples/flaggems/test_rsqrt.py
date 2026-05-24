import pytest
import torch

from .rsqrt import rsqrt, rsqrt_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_rsqrt(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu") + 0.1
    ref = torch.rsqrt(x)
    tri = rsqrt(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_rsqrt_inplace():
    x = torch.tensor([0.25, 1.0, 4.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone().rsqrt_()
    rsqrt_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
