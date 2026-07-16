import pytest
import torch

from .bitwise_and import (
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (1024,)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_and_tensor(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    ref = torch.bitwise_and(x, y)
    tri = bitwise_and_tensor(x, y)
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("shape", [(16, 256)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_and_tensor_(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    ref = torch.bitwise_and(x.clone(), y)
    x_copy = x.clone()
    bitwise_and_tensor_(x_copy, y)
    torch.testing.assert_close(x_copy, ref)


def test_bitwise_and_scalar():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cpu")
    ref = torch.bitwise_and(x, 0x3)
    tri = bitwise_and_scalar(x, 0x3)
    torch.testing.assert_close(tri, ref)


def test_bitwise_and_scalar_():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cpu")
    ref = torch.bitwise_and(x.clone(), 0x3)
    x_copy = x.clone()
    bitwise_and_scalar_(x_copy, 0x3)
    torch.testing.assert_close(x_copy, ref)


def test_bitwise_and_scalar_tensor():
    shape = (16, 256)
    dtype = torch.int32
    x = 0xF0
    y = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    ref = torch.bitwise_and(x, y)
    tri = bitwise_and_scalar_tensor(x, y)
    torch.testing.assert_close(tri, ref)
