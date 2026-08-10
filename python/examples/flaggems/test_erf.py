import pytest
import torch

from .erf import erf, erf_


def _get_tolerance(dtype):
    """Return (rtol, atol) suitable for the given dtype."""
    if dtype == torch.float16:
        return 1e-2, 1e-3
    return 1e-4, 1e-4


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_erf(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    ref = torch.erf(x)
    tri = erf(x)

    rtol, atol = _get_tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_erf_inplace(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    ref = x.clone()
    torch.erf(x, out=ref)
    tri = x.clone()
    result = erf_(tri)

    rtol, atol = _get_tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)
    assert result is tri, "erf_ must return the input tensor"
