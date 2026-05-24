import pytest
import torch

from .diff import diff


@pytest.mark.parametrize("n", [512, 1023, 1024])
def test_diff_1d(n):
    torch.manual_seed(0)
    x = torch.randn(n, dtype=torch.float32, device="cpu")

    ref = torch.diff(x)
    tri = diff(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("dim", [0, 1])
def test_diff_2d(n, dim):
    torch.manual_seed(0)
    x = torch.randn(n, n, dtype=torch.float32, device="cpu")

    ref = torch.diff(x, dim=dim)
    tri = diff(x, dim=dim)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [512, 1024])
def test_diff_n2(n):
    torch.manual_seed(0)
    x = torch.randn(n, dtype=torch.float32, device="cpu")

    ref = torch.diff(x, n=2)
    tri = diff(x, n=2)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
