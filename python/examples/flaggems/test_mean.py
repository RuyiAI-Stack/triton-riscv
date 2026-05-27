import pytest
import torch

from .mean import mean, mean_dim


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,), (16, 256)])
def test_mean(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.mean(x)
    tri = mean(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, dim", [((16, 256), 0), ((16, 256), 1), ((8, 32, 16), 2)]
)
def test_mean_dim(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.mean(x, dim=dim)
    tri = mean_dim(x, dim=dim)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
