import pytest
import torch

from .isfinite import isfinite


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.int32]
)
def test_isfinite(shape, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device="cpu")
        x[0] = float("inf")
        if shape[0] > 1:
            x[1] = float("-inf")
        if shape[0] > 2:
            x[2] = float("nan")
    else:
        x = torch.randint(-100, 100, shape, dtype=dtype, device="cpu")

    ref = torch.isfinite(x)
    tri = isfinite(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
