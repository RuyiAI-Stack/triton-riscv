import pytest
import torch

from .mul import mul, mul_


@pytest.mark.parametrize(
    "shape", [(16, 256), (4, 128), (512,), (1023,), (1024,)]
)
def test_mul_tt(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.mul(x, y)
    tri_out = mul(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape", [(16, 256), (4, 128), (512,), (1023,), (1024,)]
)
def test_mul_ts(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = 5.0

    ref_out = torch.mul(x, y)
    tri_out = mul(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape", [(16, 256), (4, 128), (512,), (1023,), (1024,)]
)
def test_mul_st(shape):
    torch.manual_seed(0)
    x = 5.0
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.mul(x, y)
    tri_out = mul(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape_A, shape_B", [((16, 256), (256,)), ((4, 1), (4, 128))]
)
def test_mul_broadcast(shape_A, shape_B):
    torch.manual_seed(0)
    x = torch.randn(shape_A, dtype=torch.float32, device="cpu")
    y = torch.randn(shape_B, dtype=torch.float32, device="cpu")

    ref_out = torch.mul(x, y)
    tri_out = mul(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_mul_complex(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.complex64, device="cpu")
    y = torch.randn(shape, dtype=torch.complex64, device="cpu")

    ref_out = torch.mul(x, y)
    tri_out = mul(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (1023,)])
def test_mul_inplace(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()

    x_ref.mul_(y)
    mul_(x, y)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
