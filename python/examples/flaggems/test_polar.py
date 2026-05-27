import pytest
import torch

from .polar import polar


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((512,), (512,)),
        ((1023,), (1023,)),
        ((1024,), (1024,)),
        ((2, 512), (2, 512)),
        ((2, 512), (512,)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_polar(shape1, shape2, dtype):
    a = torch.randn(shape1, dtype=dtype)
    b = torch.randn(shape2, dtype=dtype)

    out_triton = polar(a, b)
    out_torch = torch.polar(a, b)

    torch.testing.assert_close(out_triton, out_torch)
