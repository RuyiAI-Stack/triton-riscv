import pytest
import torch

from .ones import ones


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_ones(shape, dtype):
    tri_out = ones(shape, dtype=dtype, device="cpu")
    ref_out = torch.ones(shape, dtype=dtype, device="cpu")
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (512,)],
)
def test_ones_int(shape):
    tri_out = ones(shape, dtype=torch.int64, device="cpu")
    ref_out = torch.ones(shape, dtype=torch.int64, device="cpu")
    torch.testing.assert_close(tri_out, ref_out)
