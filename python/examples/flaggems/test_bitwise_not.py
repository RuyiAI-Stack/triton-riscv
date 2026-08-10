import pytest
import torch

from .bitwise_not import bitwise_not, bitwise_not_


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_not(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    ref = torch.bitwise_not(x)
    tri = bitwise_not(x)
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_not_inplace(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    x_clone = x.clone()
    ref = torch.bitwise_not(x_clone)
    tri = bitwise_not_(x)
    torch.testing.assert_close(tri, ref)
    torch.testing.assert_close(x, ref)
