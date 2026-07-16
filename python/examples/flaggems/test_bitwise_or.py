import pytest
import torch

from .bitwise_or import (
    bitwise_or_scalar,
    bitwise_or_scalar_,
    bitwise_or_scalar_tensor,
    bitwise_or_tensor,
    bitwise_or_tensor_,
)


@pytest.mark.parametrize("shape", [(16, 256), (1024,)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_or_tensor(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    ref = torch.bitwise_or(x, y)
    tri = bitwise_or_tensor(x, y)
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("shape", [(16, 256)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_or_tensor_(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    ref = torch.bitwise_or(x.clone(), y)
    x_copy = x.clone()
    bitwise_or_tensor_(x_copy, y)
    torch.testing.assert_close(x_copy, ref)


def test_bitwise_or_scalar():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cpu")
    ref = torch.bitwise_or(x, 0x3)
    tri = bitwise_or_scalar(x, 0x3)
    torch.testing.assert_close(tri, ref)


def test_bitwise_or_scalar_():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cpu")
    ref = torch.bitwise_or(x.clone(), 0x3)
    x_copy = x.clone()
    bitwise_or_scalar_(x_copy, 0x3)
    torch.testing.assert_close(x_copy, ref)


def test_bitwise_or_scalar_tensor():
    shape = (16, 256)
    dtype = torch.int32
    x = 0xF0
    y = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    ref = torch.bitwise_or(x, y)
    tri = bitwise_or_scalar_tensor(x, y)
    torch.testing.assert_close(tri, ref)
