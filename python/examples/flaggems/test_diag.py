import pytest
import torch

from .diag import diag


@pytest.mark.parametrize("n", [512, 1023, 1024])
def test_diag_1d_to_2d(n):
    torch.manual_seed(0)
    x = torch.randn(n, dtype=torch.float32, device="cpu")

    ref = torch.diag(x)
    tri = diag(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [128, 256])
@pytest.mark.parametrize("diagonal", [0, 1, -1, 5, -5])
def test_diag_2d_to_1d(n, diagonal):
    torch.manual_seed(0)
    x = torch.randn(n, n, dtype=torch.float32, device="cpu")

    ref = torch.diag(x, diagonal=diagonal)
    tri = diag(x, diagonal=diagonal)

    assert tri.shape == ref.shape
    if ref.numel() > 0:
        torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
