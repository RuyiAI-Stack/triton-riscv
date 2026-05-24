import pytest
import torch

from .isnan import isnan


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.int32]
)
def test_isnan(shape, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device="cpu")
        x[0] = float("nan")
    else:
        x = torch.randint(-100, 100, shape, dtype=dtype, device="cpu")

    ref = torch.isnan(x)
    tri = isnan(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
