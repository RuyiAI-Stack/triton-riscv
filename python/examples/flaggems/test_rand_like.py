import pytest
import torch

from .rand_like import rand_like


@pytest.mark.parametrize("shape", [(10,), (512,), (1023,), (1024,), (2, 512)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rand_like(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype)

    out_triton = rand_like(x)

    # For random functions, we just check properties.
    assert out_triton.shape == x.shape
    assert out_triton.dtype == x.dtype
    assert torch.all(out_triton >= 0.0)
    assert torch.all(out_triton < 1.0)
