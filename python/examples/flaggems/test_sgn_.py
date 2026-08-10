import pytest
import torch

from .sgn_ import sgn_


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_sgn_(size, dtype):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=dtype)
    x_clone = x.clone()

    ref_out = torch.sgn(x_clone.clone())
    sgn_(x)

    torch.testing.assert_close(x, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_sgn_with_zero(size, dtype):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=dtype)
    x[10] = 0.0
    ref_out = torch.sgn(x.clone())
    sgn_(x)

    torch.testing.assert_close(x, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_sgn_integer(size):
    torch.manual_seed(0)
    x = torch.randint(-100, 100, (size,), device="cpu", dtype=torch.int32)
    ref_out = torch.sgn(x.clone())
    sgn_(x)

    torch.testing.assert_close(x, ref_out)
