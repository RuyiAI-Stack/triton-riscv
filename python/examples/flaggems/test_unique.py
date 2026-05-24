import pytest
import torch

from .unique import _unique2


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_unique_values(size):
    torch.manual_seed(0)
    x = torch.randint(0, size // 4, (size,), dtype=torch.float32, device="cpu")

    ref_values = torch.unique(x)
    tri_values, _, _ = _unique2(x)

    torch.testing.assert_close(tri_values, ref_values, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_unique_return_inverse(size):
    torch.manual_seed(0)
    x = torch.randint(0, size // 4, (size,), dtype=torch.float32, device="cpu")

    ref_values, ref_inverse = torch.unique(x, return_inverse=True)
    tri_values, tri_inverse, _ = _unique2(x, return_inverse=True)

    torch.testing.assert_close(tri_values, ref_values, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_inverse, ref_inverse)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_unique_return_counts(size):
    torch.manual_seed(0)
    x = torch.randint(0, size // 4, (size,), dtype=torch.float32, device="cpu")

    ref_values, ref_counts = torch.unique(x, return_counts=True)
    tri_values, _, tri_counts = _unique2(x, return_counts=True)

    torch.testing.assert_close(tri_values, ref_values, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_counts, ref_counts)
