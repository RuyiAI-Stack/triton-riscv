import pytest
import torch

from .median import median, median_dim


@pytest.mark.parametrize("shape", [(16,), (8, 4), (2, 3, 4)])
def test_median_flat(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float32)
    ref = torch.median(x)
    tri = median(x)
    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape, dim", [((4, 8), 0), ((4, 8), 1)])
def test_median_dim(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float32)
    ref_vals, ref_idxs = torch.median(x, dim=dim)
    tri_vals, tri_idxs = median_dim(x, dim=dim)
    torch.testing.assert_close(tri_vals, ref_vals, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_idxs, ref_idxs)


@pytest.mark.parametrize("rows,dtype", [(16, torch.float16), (17, torch.float32)])
def test_median_dim_strided_row_offsets(rows, dtype):
    x = torch.arange(rows * 4, dtype=dtype).reshape(rows, 4, 1)
    ref_values, ref_indices = torch.median(x, dim=1)
    tri_values, tri_indices = median_dim(x, dim=1)
    torch.testing.assert_close(tri_values, ref_values, rtol=0, atol=0)
    torch.testing.assert_close(tri_indices, ref_indices)
