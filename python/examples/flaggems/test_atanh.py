import pytest
import torch

from .atanh import atanh, atanh_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_atanh(size):
    x = torch.linspace(-0.8, 0.8, size, dtype=torch.float32, device="cpu")

    out = atanh(x)
    ref = torch.atanh(x)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_atanh_inplace():
    x = torch.tensor([-0.25, 0.0, 0.25], dtype=torch.float32, device="cpu")
    ref = torch.atanh(x)

    atanh_(x)

    torch.testing.assert_close(x, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_atanh_(shape, dtype):
    torch.manual_seed(0)
    # atanh expects values in (-1, 1)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 1.8 - 0.9
    x2 = x.clone()

    ref_out = torch.atanh(x)
    tri_out = atanh_(x2)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x2, ref_out, rtol=1e-3, atol=1e-3)


def test_atanh_integral_promotes_like_torch():
    x = torch.tensor([-1, 0, 1], dtype=torch.int32, device="cpu")

    out = atanh(x)
    ref = torch.atanh(x)

    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref)


def test_atanh_inplace_rejects_integral_input():
    x = torch.tensor([0, 1], dtype=torch.int32, device="cpu")

    with pytest.raises(RuntimeError):
        atanh_(x)


def test_atanh_inplace_supports_noncontiguous_input():
    x = torch.linspace(-0.8, 0.8, 12, dtype=torch.float32, device="cpu").view(3, 4).t()
    ref = torch.atanh(x)

    returned = atanh_(x)

    assert returned is x
    assert not x.is_contiguous()
    torch.testing.assert_close(x, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_atanh_complex(dtype):
    x = torch.tensor([0.5 + 0.25j, -0.2 + 0.1j], dtype=dtype, device="cpu")

    out = atanh(x)
    ref = torch.atanh(x)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_atanh_inplace_complex(dtype):
    x = torch.tensor([0.5 + 0.25j, -0.2 + 0.1j], dtype=dtype, device="cpu")
    ref = torch.atanh(x)

    returned = atanh_(x)

    assert returned is x
    torch.testing.assert_close(x, ref, rtol=1e-5, atol=1e-5)
