import pytest
import torch

from .min import min, min_dim


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,), (16, 256)])
def test_min(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.min(x)
    tri = min(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, dim", [((16, 256), 0), ((16, 256), 1), ((8, 32, 16), 2)]
)
def test_min_dim(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_vals, ref_indices = torch.min(x, dim=dim)
    tri = min_dim(x, dim=dim)

    torch.testing.assert_close(tri.values, ref_vals, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri.indices, ref_indices, rtol=0, atol=0)
