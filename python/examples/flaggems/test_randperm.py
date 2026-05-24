import pytest
import torch

from .randperm import randperm


@pytest.mark.parametrize("n", [10, 1024, 2050])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_randperm(n, dtype):
    torch.manual_seed(0)

    out_triton = randperm(n, dtype=dtype)
    out_torch = torch.randperm(n, dtype=dtype)

    assert out_triton.shape == out_torch.shape
    assert out_triton.dtype == out_torch.dtype

    # Verify it contains all numbers from 0 to n-1
    triton_sorted = torch.sort(out_triton)[0]
    torch_sorted = torch.arange(n, dtype=dtype)
    torch.testing.assert_close(triton_sorted, torch_sorted)
