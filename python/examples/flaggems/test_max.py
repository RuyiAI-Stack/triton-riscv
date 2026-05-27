import pytest
import torch

from .max import max, max_dim


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,), (16, 256)])
def test_max(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.max(x)
    tri = max(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, dim", [((16, 256), 0), ((16, 256), 1), ((8, 32, 16), 2)]
)
def test_max_dim(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_vals, ref_indices = torch.max(x, dim=dim)
    tri = max_dim(x, dim=dim)

    torch.testing.assert_close(tri.values, ref_vals, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri.indices, ref_indices, rtol=0, atol=0)
