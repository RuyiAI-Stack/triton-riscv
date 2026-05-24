import pytest
import torch

from .bincount import bincount


@pytest.mark.parametrize("size", [16, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bincount_basic(size, dtype):
    torch.manual_seed(0)
    inp = torch.randint(0, 10, (size,), dtype=dtype, device="cpu")
    ref = torch.bincount(inp)
    tri = bincount(inp)
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("size", [16, 128])
def test_bincount_weights(size):
    torch.manual_seed(0)
    inp = torch.randint(0, 10, (size,), dtype=torch.int32, device="cpu")
    weights = torch.randn(size, dtype=torch.float32, device="cpu")
    ref = torch.bincount(inp, weights=weights)
    tri = bincount(inp, weights=weights)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_bincount_minlength():
    inp = torch.tensor([1, 1, 3], dtype=torch.int32, device="cpu")
    ref = torch.bincount(inp, minlength=10)
    tri = bincount(inp, minlength=10)
    torch.testing.assert_close(tri, ref)


def test_bincount_empty():
    inp = torch.tensor([], dtype=torch.int32, device="cpu")
    ref = torch.bincount(inp)
    tri = bincount(inp)
    torch.testing.assert_close(tri, ref)


def test_bincount_uint8():
    inp = torch.tensor([0, 2, 5, 2, 0], dtype=torch.uint8, device="cpu")
    ref = torch.bincount(inp)
    tri = bincount(inp)
    torch.testing.assert_close(tri, ref)
