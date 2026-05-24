import pytest
import torch

from .asinh_ import asinh_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_asinh_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref = x.clone()
    torch.asinh_(ref)

    tri = x.clone()
    out = asinh_(tri)

    assert out is tri, "asinh_ should return the same tensor"
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_asinh_inplace_no_ref(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x_ref = x.clone()

    x_ref.asinh_()
    asinh_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


def test_asinh_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32)
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = x_sliced.clone()
    torch.asinh_(ref)

    tri = x_sliced.clone()
    out = asinh_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_asinh_inplace_other_float(dtype):
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=dtype)

    ref = x.clone()
    torch.asinh_(ref)

    tri = x.clone()
    out = asinh_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)
