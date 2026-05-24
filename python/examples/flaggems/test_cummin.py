import pytest
import torch

from .cummin import cummin


@pytest.mark.parametrize("shape", [(16, 256)])
@pytest.mark.parametrize("dim", [0, 1])
def test_cummin(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_val, ref_idx = torch.cummin(x, dim=dim)
    tri_val, tri_idx = cummin(x, dim=dim)

    torch.testing.assert_close(tri_val, ref_val, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_idx, ref_idx)


@pytest.mark.parametrize("shape", [(4, 32)])
@pytest.mark.parametrize("dim", [0, 1])
def test_cummin_int(shape, dim):
    torch.manual_seed(0)
    x = torch.randint(-100, 100, shape, device="cpu", dtype=torch.int32)

    ref_val, ref_idx = torch.cummin(x, dim=dim)
    tri_val, tri_idx = cummin(x, dim=dim)

    torch.testing.assert_close(tri_val, ref_val)
    torch.testing.assert_close(tri_idx, ref_idx)


@pytest.mark.parametrize("shape", [(2, 4, 8)])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_cummin_3d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_val, ref_idx = torch.cummin(x, dim=dim)
    tri_val, tri_idx = cummin(x, dim=dim)

    torch.testing.assert_close(tri_val, ref_val, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_idx, ref_idx)
