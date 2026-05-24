import pytest
import torch

from .randn import randn


@pytest.mark.parametrize("shape", [(10,), (512,), (1023,), (1024,), (2, 512)])
def test_randn(shape):
    torch.manual_seed(0)

    out_triton = randn(shape, dtype=torch.float32)

    # For random functions, we check basic properties.
    assert out_triton.shape == shape
    assert out_triton.dtype == torch.float32
