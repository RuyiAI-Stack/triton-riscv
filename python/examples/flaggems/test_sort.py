import pytest
import torch

from .sort import sort, sort_stable


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_sort(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")

    ref_sorted, ref_indices = torch.sort(x, dim=-1)
    tri_sorted, tri_indices = sort(x, dim=-1)

    torch.testing.assert_close(tri_sorted, ref_sorted, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_indices, ref_indices)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_sort_descending(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")

    ref_sorted, ref_indices = torch.sort(x, dim=-1, descending=True)
    tri_sorted, tri_indices = sort(x, dim=-1, descending=True)

    torch.testing.assert_close(tri_sorted, ref_sorted, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_indices, ref_indices)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_sort_stable(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")

    ref_sorted, ref_indices = torch.sort(x, dim=-1, stable=True)
    tri_sorted, tri_indices = sort_stable(x, stable=True, dim=-1)

    torch.testing.assert_close(tri_sorted, ref_sorted, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_indices, ref_indices)
