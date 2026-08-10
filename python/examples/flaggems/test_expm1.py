import pytest
import torch

from .expm1 import expm1, expm1_, expm1_out


def _get_tolerance(dtype):
    """Return (rtol, atol) suitable for the given dtype."""
    if dtype == torch.float16:
        return 1e-2, 1e-3
    return 1e-4, 1e-4


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_expm1(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    ref = torch.expm1(x)
    tri = expm1(x)

    rtol, atol = _get_tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_expm1_inplace(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    ref = x.clone()
    torch.expm1(x, out=ref)
    tri = x.clone()
    result = expm1_(tri)

    rtol, atol = _get_tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)
    assert result is tri, "expm1_ must return the input tensor"


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_expm1_out(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    ref = torch.expm1(x)
    out = torch.empty_like(x)
    result = expm1_out(x, out)

    rtol, atol = _get_tolerance(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    assert result is out, "expm1_out must return the output tensor"
