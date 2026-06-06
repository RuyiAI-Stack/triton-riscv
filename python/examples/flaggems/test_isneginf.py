import pytest
import torch

from .isneginf import isneginf, isneginf_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_isneginf(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    x[0] = float("-inf")

    ref = torch.isneginf(x)
    tri = isneginf(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_isneginf_out(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    x[0] = float("-inf")

    ref = torch.isneginf(x)

    out = torch.empty(shape, dtype=torch.bool, device="cpu")
    tri = isneginf_out(x, out=out)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
