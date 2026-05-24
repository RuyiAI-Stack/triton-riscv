import pytest
import torch

from .repeat import repeat


@pytest.mark.parametrize(
    "shape, sizes",
    [
        ((5,), (2,)),
        ((5,), (2, 3)),
        ((2, 3), (2, 2)),
        ((2, 1, 3), (1, 4, 2)),
        ((1, 2, 1, 3), (2, 1, 4, 1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_repeat(shape, sizes, dtype):
    x = torch.randn(shape).to(dtype)

    out_triton = repeat(x, sizes)
    out_torch = x.repeat(sizes)

    torch.testing.assert_close(out_triton, out_torch)
