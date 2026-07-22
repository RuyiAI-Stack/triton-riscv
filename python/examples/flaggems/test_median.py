import pytest
import torch

from .median import median, median_dim, median_dim_values, median_out


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


def test_median_out():
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32)
    ref = torch.median(x)
    out = torch.empty((), dtype=x.dtype)

    returned = median_out(x, out=out)

    assert returned is out
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_median_dim_values():
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32)
    ref_values, ref_indices = torch.median(x, dim=1)
    values = torch.empty_like(ref_values)
    indices = torch.empty_like(ref_indices)

    returned = median_dim_values(x, dim=1, values=values, indices=indices)

    assert returned.values is values
    assert returned.indices is indices
    torch.testing.assert_close(values, ref_values, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(indices, ref_indices)
