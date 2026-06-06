import pytest
import torch

from .logical_xor import logical_xor


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_logical_xor(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 2, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 2, shape, dtype=dtype, device="cpu")

    ref = torch.logical_xor(x, y)
    tri = logical_xor(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_logical_xor_bool():
    x = torch.tensor([True, False, True, False], dtype=torch.bool, device="cpu")
    y = torch.tensor([True, True, False, False], dtype=torch.bool, device="cpu")
    ref = torch.logical_xor(x, y)
    tri = logical_xor(x, y)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logical_xor_int32(shape):
    torch.manual_seed(0)
    x = torch.randint(0, 5, shape, dtype=torch.int32, device="cpu")
    y = torch.randint(0, 5, shape, dtype=torch.int32, device="cpu")

    ref = torch.logical_xor(x, y)
    tri = logical_xor(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
