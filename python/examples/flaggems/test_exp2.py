import pytest
import torch

from .exp2 import exp2, exp2_


def _get_tolerance(dtype):
    """Return (rtol, atol) suitable for the given dtype."""
    if dtype == torch.float16:
        return 1e-2, 1e-3
    return 1e-4, 1e-4


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_exp2(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    ref = torch.exp2(x)
    tri = exp2(x)

    rtol, atol = _get_tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64]
)
def test_exp2_inplace(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    ref = x.clone()
    torch.exp2(x, out=ref)
    tri = x.clone()
    result = exp2_(tri)

    rtol, atol = _get_tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)
    assert result is tri, "exp2_ must return the input tensor"
