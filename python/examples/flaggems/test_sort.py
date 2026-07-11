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


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_repeated_values_and_padding_boundary(descending):
    max_i32 = torch.iinfo(torch.int32).max
    min_i32 = torch.iinfo(torch.int32).min
    x = torch.tensor(
        [max_i32, 7, max_i32, 7, min_i32, min_i32, 7],
        dtype=torch.int32,
    )

    ref_values, ref_indices = torch.sort(x, stable=True, descending=descending)
    tri_values, tri_indices = sort_stable(x, stable=True, descending=descending)

    torch.testing.assert_close(tri_values, ref_values)
    torch.testing.assert_close(tri_indices, ref_indices)
    assert (tri_indices < x.numel()).all()
