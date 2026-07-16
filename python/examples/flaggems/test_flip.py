import pytest
import torch

from .flip import flip


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_flip_1d(size, dtype):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=dtype)

    ref_out = torch.flip(x, dims=[0])
    tri_out = flip(x, dims=[0])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(8, 16), (16, 8), (32, 32)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_flip_2d(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=dtype)

    ref_out = torch.flip(x, dims=[0, 1])
    tri_out = flip(x, dims=[0, 1])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(4, 8, 16), (8, 4, 16)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_flip_3d(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=dtype)

    ref_out = torch.flip(x, dims=[0, 2])
    tri_out = flip(x, dims=[0, 2])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_flip_empty_dims(size, dtype):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=dtype)

    ref_out = torch.flip(x, dims=[])
    tri_out = flip(x, dims=[])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
