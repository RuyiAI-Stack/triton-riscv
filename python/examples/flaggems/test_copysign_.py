import pytest
import torch

from .copysign_ import copysign_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_copysign_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)

    ref = torch.copysign(x, y)

    tri = x.clone()
    out = copysign_(tri, y)

    assert out is tri, "copysign_ should return the same tensor"
    torch.testing.assert_close(tri, ref)


def test_copysign_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32)
    y = torch.randn(32, 64, device="cpu", dtype=torch.float32)
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = torch.copysign(x_sliced, y[::2, ::2])

    tri = x_sliced.clone()
    out = copysign_(tri, y[::2, ::2])

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_copysign_inplace_scalar_other():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32)

    ref = torch.copysign(x, -1.0)

    tri = x.clone()
    out = copysign_(tri, -1.0)

    assert out is tri
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_copysign_inplace_other_float(dtype):
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.float32).to(dtype)
    y = torch.randn(128, device="cpu", dtype=torch.float32).to(dtype)

    ref = torch.copysign(x, y)

    tri = x.clone()
    out = copysign_(tri, y)

    assert out is tri
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
    ],
)
def test_copysign__broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randn(shape_a)
    b = torch.randn(shape_b)
    ref = torch.copysign(a, b)

    tri = a.clone()
    out = copysign_(tri, b)

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_copysign_bf16():
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.bfloat16)
    y = torch.randn(128, device="cpu", dtype=torch.bfloat16)

    ref = torch.copysign(x, y)

    tri = x.clone()
    out = copysign_(tri, y)

    assert out is tri
    torch.testing.assert_close(tri, ref)
