import pytest
import torch

from .unique_consecutive import unique_consecutive


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_unique_consecutive_values(size):
    torch.manual_seed(0)
    # Create input with some consecutive duplicates
    x = torch.repeat_interleave(
        torch.randint(
            0, size // 4, (size // 2,), dtype=torch.float32, device="cpu"
        ),
        2,
    )[:size]

    ref_values = torch.unique_consecutive(x)
    tri_values, _, _ = unique_consecutive(x)

    torch.testing.assert_close(tri_values, ref_values, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_unique_consecutive_return_inverse(size):
    torch.manual_seed(0)
    x = torch.repeat_interleave(
        torch.randint(
            0, size // 4, (size // 2,), dtype=torch.float32, device="cpu"
        ),
        2,
    )[:size]

    ref_values, ref_inverse = torch.unique_consecutive(x, return_inverse=True)
    tri_values, tri_inverse, _ = unique_consecutive(x, return_inverse=True)

    torch.testing.assert_close(tri_values, ref_values, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_inverse, ref_inverse)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_unique_consecutive_return_counts(size):
    torch.manual_seed(0)
    x = torch.repeat_interleave(
        torch.randint(
            0, size // 4, (size // 2,), dtype=torch.float32, device="cpu"
        ),
        2,
    )[:size]

    ref_values, ref_counts = torch.unique_consecutive(x, return_counts=True)
    tri_values, _, tri_counts = unique_consecutive(x, return_counts=True)

    torch.testing.assert_close(tri_values, ref_values, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_counts, ref_counts)
