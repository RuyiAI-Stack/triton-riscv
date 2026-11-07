import pytest
import torch

from .logical_not import logical_not, logical_not_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_logical_not(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 2, shape, dtype=dtype, device="cpu")

    ref = torch.logical_not(x)
    tri = logical_not(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_logical_not_(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu") > 0
    if dtype != torch.bool:
        x = x.to(dtype)
    x2 = x.clone()

    ref_out = x.clone().logical_not_()
    tri_out = logical_not_(x2)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x2, ref_out, rtol=1e-3, atol=1e-3)
