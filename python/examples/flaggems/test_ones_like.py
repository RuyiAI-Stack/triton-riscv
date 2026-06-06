import pytest
import torch

from .ones_like import ones_like


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_ones_like(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cpu")
    tri_out = ones_like(x)
    ref_out = torch.ones_like(x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,)],
)
def test_ones_like_int(shape):
    x = torch.randint(0, 100, shape, device="cpu")
    tri_out = ones_like(x, dtype=torch.int64)
    ref_out = torch.ones_like(x, dtype=torch.int64)
    torch.testing.assert_close(tri_out, ref_out)
