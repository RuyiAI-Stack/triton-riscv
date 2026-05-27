import pytest
import torch

from .rand import rand


@pytest.mark.parametrize("shape", [(10,), (512,), (1023,), (1024,), (2, 512)])
def test_rand(shape):
    torch.manual_seed(0)

    out_triton = rand(shape, dtype=torch.float32)

    # For random functions, we can't assert exact values easily against PyTorch
    # without matching the exact PRNG state. So we just check properties.
    assert out_triton.shape == shape
    assert out_triton.dtype == torch.float32
    assert torch.all(out_triton >= 0.0)
    assert torch.all(out_triton < 1.0)
