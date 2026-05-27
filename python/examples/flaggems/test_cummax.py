import pytest
import torch

from .cummax import cummax


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((16, 256), 0),
        ((16, 256), 1),
        ((4, 128), -1),
        ((2, 3, 4), 1),
        # Test sizes around and above 1024
        ((512,), 0),
        ((1023,), 0),
        ((1024,), 0),
        ((2048,), 0),
    ],
)
def test_cummax(shape, dim):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn(shape, dtype=torch.float32, device=device)

    ref_out, ref_indices = torch.cummax(x, dim)
    tri_out, tri_indices = cummax(x, dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_indices, ref_indices, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((16, 256), 0),
        ((16, 256), 1),
    ],
)
def test_cummax_int(shape, dim):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randint(-100, 100, shape, dtype=torch.int32, device=device)

    ref_out, ref_indices = torch.cummax(x, dim)
    tri_out, tri_indices = cummax(x, dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_indices, ref_indices, rtol=1e-3, atol=1e-3)
