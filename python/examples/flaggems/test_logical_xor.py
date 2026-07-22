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


@pytest.mark.parametrize("x_shape, y_shape", [((16, 1), (1, 257)), ((1,), (1023,))])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.float32])
def test_logical_xor_broadcast(x_shape, y_shape, dtype):
    torch.manual_seed(0)
    if dtype is torch.float32:
        x = torch.randn(x_shape, dtype=dtype, device="cpu")
        y = torch.randn(y_shape, dtype=dtype, device="cpu")
    else:
        x = torch.randint(0, 2, x_shape, dtype=dtype, device="cpu")
        y = torch.randint(0, 2, y_shape, dtype=dtype, device="cpu")

    tri = logical_xor(x, y)
    ref = torch.logical_xor(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_logical_xor_non_contiguous_input():
    torch.manual_seed(0)
    x = torch.randint(0, 2, (32, 64), dtype=torch.int32, device="cpu")[:, ::2]
    y = torch.randint(0, 2, x.shape, dtype=torch.int32, device="cpu")

    tri = logical_xor(x, y)
    ref = torch.logical_xor(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_logical_xor_empty_tensor():
    x = torch.empty((0, 17), dtype=torch.bool, device="cpu")
    y = torch.empty((1, 17), dtype=torch.bool, device="cpu")

    tri = logical_xor(x, y)
    ref = torch.logical_xor(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_logical_xor_float_truthiness_matches_torch():
    x = torch.tensor([-2.0, -0.0, 0.0, 0.5], device="cpu")
    y = torch.tensor([0.0, -1.0, 2.0, 0.0], device="cpu")

    tri = logical_xor(x, y)
    ref = torch.logical_xor(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
