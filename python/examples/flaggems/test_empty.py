import torch

from .empty import empty


def test_empty_metadata_only():
    # The storage is deliberately uninitialized: only metadata is pinned.
    t = empty((2, 3))
    assert t.shape == (2, 3)
    assert t.dtype == torch.get_default_dtype()
    assert t.is_contiguous()


def test_empty_dtype_override():
    t = empty((4,), dtype=torch.int64)
    assert t.dtype == torch.int64


def test_empty_memory_format_channels_last():
    t = empty((2, 3, 4, 4), dtype=torch.float64, memory_format=torch.channels_last)
    assert t.stride() == (48, 1, 12, 3)  # channels_last: (H*W*C, 1, W*C, C)


def test_empty_is_fresh_allocation():
    a = empty((1024,))
    b = empty((1024,))
    assert a.data_ptr() != b.data_ptr()
