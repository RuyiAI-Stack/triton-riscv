import pytest
import torch

from .bitwise_xor import (
    bitwise_xor_scalar,
    bitwise_xor_scalar_,
    bitwise_xor_scalar_tensor,
    bitwise_xor_tensor,
    bitwise_xor_tensor_,
)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (1024,), (1023,)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_xor_tensor(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    ref = torch.bitwise_xor(x, y)
    tri = bitwise_xor_tensor(x, y)
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("shape", [(16, 256)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_xor_tensor_(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 255, shape, dtype=dtype, device="cpu")
    ref = torch.bitwise_xor(x.clone(), y)
    x_copy = x.clone()
    out = bitwise_xor_tensor_(x_copy, y)
    assert out is x_copy
    torch.testing.assert_close(x_copy, ref)


def test_bitwise_xor_scalar():
    torch.manual_seed(0)
    x = torch.randint(0, 255, (1024,), dtype=torch.int32, device="cpu")
    ref = torch.bitwise_xor(x, 0x5)
    tri = bitwise_xor_scalar(x, 0x5)
    torch.testing.assert_close(tri, ref)


def test_bitwise_xor_scalar_():
    torch.manual_seed(0)
    x = torch.randint(0, 255, (1024,), dtype=torch.int32, device="cpu")
    ref = torch.bitwise_xor(x.clone(), 0x5)
    x_copy = x.clone()
    out = bitwise_xor_scalar_(x_copy, 0x5)
    assert out is x_copy
    torch.testing.assert_close(x_copy, ref)


def test_bitwise_xor_scalar_tensor():
    torch.manual_seed(0)
    y = torch.randint(0, 255, (1024,), dtype=torch.int32, device="cpu")
    ref = torch.bitwise_xor(0xF0, y)
    tri = bitwise_xor_scalar_tensor(0xF0, y)
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_bitwise_xor_tensor_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randint(0, 255, shape_a, dtype=torch.int32)
    b = torch.randint(0, 255, shape_b, dtype=torch.int32)

    ref_out = torch.bitwise_xor(a, b)
    tri_out = bitwise_xor_tensor(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_bitwise_xor_negative_values():
    torch.manual_seed(0)
    x = torch.randint(-128, 128, (1024,), dtype=torch.int32, device="cpu")
    y = torch.randint(-128, 128, (1024,), dtype=torch.int32, device="cpu")

    ref = torch.bitwise_xor(x, y)
    tri = bitwise_xor_tensor(x, y)

    torch.testing.assert_close(tri, ref)


def test_bitwise_xor_inplace_small_out_raises():
    # Broadcastable shapes with a smaller in-place target must raise instead
    # of launching with the broadcast numel over a smaller buffer.
    a = torch.zeros(16, 1, dtype=torch.int32)
    b = torch.ones(16, 256, dtype=torch.int32)
    with pytest.raises(ValueError):
        bitwise_xor_tensor_(a, b)


def test_bitwise_xor_mixed_int_promotes():
    # torch promotes (int32, int64) -> int64 without truncating.
    x = torch.tensor([5, -1], dtype=torch.int32)  # -1 == 0xFFFFFFFF bits
    y = torch.tensor([3, 1], dtype=torch.int64)
    out = bitwise_xor_tensor(x, y)
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, torch.bitwise_xor(x, y))
