import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def scalar_masked_load(X, Indices, Y, Other):
    pid = tl.program_id(0)
    index = tl.load(Indices + pid)
    other = tl.load(Other)
    value = tl.load(X + index, mask=index >= 0, other=other)
    tl.store(Y + pid, value)


@triton.jit
def scalar_masked_store(X, Indices, Y):
    pid = tl.program_id(0)
    index = tl.load(Indices + pid)
    value = tl.load(X + pid)
    tl.store(Y + index, value, mask=index >= 0)


@pytest.mark.parametrize(
    "precision", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_scalar_load_mask_and_other(precision):
    # A masked negative index points to a valid sentinel: a wrong load is observable
    # without relying on a crash or on the allocator's surrounding memory.
    storage = torch.tensor([123, 10, 20, 30], dtype=precision)
    indices = torch.tensor([-1, 0, 2], dtype=torch.int32)
    result = torch.empty(3, dtype=precision)
    other = torch.tensor(-7, dtype=precision)
    scalar_masked_load[(3,)](storage[1:], indices, result, other)
    torch.testing.assert_close(
        result, torch.tensor([-7, 10, 30], dtype=precision), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "precision", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_scalar_store_mask_preserves_sentinel(precision):
    storage = torch.tensor([123, 10, 20, 30], dtype=precision)
    indices = torch.tensor([-1, 0, 2], dtype=torch.int32)
    values = torch.tensor([99, 41, 52], dtype=precision)
    scalar_masked_store[(3,)](values, indices, storage[1:])
    torch.testing.assert_close(
        storage, torch.tensor([123, 41, 20, 52], dtype=precision), rtol=0, atol=0
    )
