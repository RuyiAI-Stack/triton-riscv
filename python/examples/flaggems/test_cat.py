import pytest
import torch

from .cat import cat, cat_out


@pytest.mark.parametrize("dim", [0, 1])
def test_cat_2d(dim):
    torch.manual_seed(0)
    if dim == 0:
        a = torch.randn(4, 8, dtype=torch.float32, device="cpu")
        b = torch.randn(6, 8, dtype=torch.float32, device="cpu")
    else:
        a = torch.randn(4, 8, dtype=torch.float32, device="cpu")
        b = torch.randn(4, 6, dtype=torch.float32, device="cpu")
    ref = torch.cat([a, b], dim=dim)
    tri = cat([a, b], dim=dim)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_cat_3d(dim):
    torch.manual_seed(0)
    a = torch.randn(2, 3, 4, dtype=torch.float32, device="cpu")
    b = torch.randn(2, 3, 4, dtype=torch.float32, device="cpu")
    ref = torch.cat([a, b], dim=dim)
    tri = cat([a, b], dim=dim)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_cat_three_tensors():
    torch.manual_seed(0)
    a = torch.randn(3, 4, dtype=torch.float32, device="cpu")
    b = torch.randn(5, 4, dtype=torch.float32, device="cpu")
    c = torch.randn(2, 4, dtype=torch.float32, device="cpu")
    ref = torch.cat([a, b, c])
    tri = cat([a, b, c])
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_cat_single():
    a = torch.randn(4, 4, dtype=torch.float32, device="cpu")
    ref = torch.cat([a])
    tri = cat([a])
    torch.testing.assert_close(tri, ref)


def test_cat_empty():
    a = torch.randn(0, 4, dtype=torch.float32, device="cpu")
    b = torch.randn(3, 4, dtype=torch.float32, device="cpu")
    ref = torch.cat([a, b])
    tri = cat([a, b])
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_cat_out():
    a = torch.randn(4, 8, dtype=torch.float32, device="cpu")
    b = torch.randn(6, 8, dtype=torch.float32, device="cpu")
    out = torch.empty(10, 8, dtype=torch.float32, device="cpu")
    ref = torch.cat([a, b])
    cat_out([a, b], out=out)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
