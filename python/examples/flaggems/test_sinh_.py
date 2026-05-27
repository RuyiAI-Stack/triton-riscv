import pytest
import torch

from .sinh_ import sinh_


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sinh_(size, dtype):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=dtype)
    x_ref = x.clone()

    x_ref.sinh_()
    sinh_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
