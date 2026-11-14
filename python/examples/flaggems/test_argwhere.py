import pytest
import torch

from .argwhere import argwhere


def _make_sparse(shape, zero_frac=0.6):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)
    x[torch.rand_like(x) < zero_frac] = 0.0
    return x


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_argwhere_1d(size):
    x = _make_sparse((size,))

    ref = torch.argwhere(x)
    tri = argwhere(x)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref)


def test_argwhere_2d():
    x = _make_sparse((16, 64))

    ref = torch.argwhere(x)
    tri = argwhere(x)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref)


def test_argwhere_3d():
    x = _make_sparse((4, 8, 32))

    ref = torch.argwhere(x)
    tri = argwhere(x)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref)


def test_argwhere_all_zero():
    x = torch.zeros(128, device="cpu", dtype=torch.float32)

    ref = torch.argwhere(x)
    tri = argwhere(x)

    assert tri.shape == (0, 1)
    torch.testing.assert_close(tri, ref)


def test_argwhere_no_zero():
    x = torch.randn(128, device="cpu", dtype=torch.float32) + 10.0

    ref = torch.argwhere(x)
    tri = argwhere(x)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref)


def test_argwhere_int():
    torch.manual_seed(0)
    x = torch.randint(0, 3, (256,), dtype=torch.int32)

    ref = torch.argwhere(x)
    tri = argwhere(x)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref)


def test_argwhere_multiblock():
    # 5000 elements span multiple _BLOCK_SIZE=2048 scans: exercises the
    # cross-block carry and the tail-block padding.
    x = _make_sparse((5000,))
    torch.testing.assert_close(argwhere(x), torch.argwhere(x))
