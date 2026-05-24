import pytest
import torch

from .randn_like import randn_like


@pytest.mark.parametrize("shape", [(10,), (512,), (1023,), (1024,), (2, 512)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_randn_like(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype)

    out_triton = randn_like(x)

    # For random functions, we just check properties.
    assert out_triton.shape == x.shape
    assert out_triton.dtype == x.dtype
