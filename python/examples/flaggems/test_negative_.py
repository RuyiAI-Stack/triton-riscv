import pytest
import torch

from .negative_ import negative_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_negative_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref = x.clone()
    torch.negative_(ref)

    tri = x.clone()
    out = negative_(tri)

    assert out is tri, "negative_ should return the same tensor"
    torch.testing.assert_close(tri, ref.t())


def test_negative_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32)
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = x_sliced.clone()
    torch.negative_(ref)

    # .t() keeps the view genuinely non-contiguous (a .clone() of a
    # strided view re-packs contiguous, which would only exercise
    # the fast path).
    tri = x_sliced.t()
    assert not tri.is_contiguous()
    out = negative_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_negative_inplace_other_float(dtype):
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=dtype)

    ref = x.clone()
    torch.negative_(ref)

    tri = x.clone()
    out = negative_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t())


def test_negative_bf16():
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.bfloat16)

    ref = torch.negative(x)

    tri = x.clone()
    out = negative_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t())


def test_negative_empty():
    x = torch.empty(0)
    negative_(x)
    assert x.numel() == 0
