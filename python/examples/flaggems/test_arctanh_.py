import pytest
import torch

from .arctanh_ import arctanh_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_arctanh_inplace(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device="cpu") * 0.5  # keep in (-1, 1)
    x_ref = x.clone()

    arctanh_(x)
    ref = torch.arctanh(x_ref)

    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)


def test_arctanh_inplace_contiguous_required():
    x = torch.rand(5, 5, device="cpu", dtype=torch.float32).T
    with pytest.raises(ValueError, match="contiguous"):
        arctanh_(x)


def test_arctanh_inplace_float_only():
    x = torch.randint(0, 1, (10,), dtype=torch.int32, device="cpu")
    with pytest.raises(TypeError, match="floating point"):
        arctanh_(x)
