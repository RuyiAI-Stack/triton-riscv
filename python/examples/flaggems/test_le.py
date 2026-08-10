import pytest
import torch

from .le import le, le_scalar


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.int32]
)
def test_le(shape, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device="cpu")
        y = torch.randn(shape, dtype=dtype, device="cpu")
    else:
        x = torch.randint(-100, 100, shape, dtype=dtype, device="cpu")
        y = torch.randint(-100, 100, shape, dtype=dtype, device="cpu")

    ref = torch.le(x, y)
    tri = le(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.int32]
)
def test_le_scalar(dtype):
    if dtype.is_floating_point:
        x = torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=dtype, device="cpu")
        scalar = 1.0
    else:
        x = torch.tensor([0, 1, 2, 0], dtype=dtype, device="cpu")
        scalar = 1
    tri = le_scalar(x, scalar)
    ref = torch.le(x, scalar)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
