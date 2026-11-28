import importlib

import pytest
import torch

from .split_with_sizes_copy import split_with_sizes_copy


def test_split_with_sizes_copy_matches_aten():
    x = torch.arange(2 * 7 * 3, dtype=torch.float64).reshape(2, 7, 3)
    for sizes, dim in [([2, 5], 1), ([1, 1, 5], 1), ([1, 1], 0), ([3], 2), ([7], -2)]:
        chunks = split_with_sizes_copy(x, sizes, dim)
        ref = torch.ops.aten.split_with_sizes_copy(x, sizes, dim)
        assert len(chunks) == len(ref)
        for c, r in zip(chunks, ref):
            torch.testing.assert_close(c, r, rtol=0, atol=0, msg=f"{sizes},{dim}")


def test_split_with_sizes_copy_sum_mismatch_asserts():
    x = torch.arange(6, dtype=torch.float64)
    with pytest.raises(AssertionError, match="Invalid split_sizes"):
        split_with_sizes_copy(x, [2, 5], 0)


def test_split_with_sizes_copy_strided_input():
    x = torch.randn(4, 8).t()  # (8, 4) non-contiguous
    chunks = split_with_sizes_copy(x, [3, 5], 0)
    ref = torch.ops.aten.split_with_sizes_copy(x, [3, 5], 0)
    for c, r in zip(chunks, ref):
        torch.testing.assert_close(c, r, rtol=0, atol=0)


def test_split_with_sizes_copy_zero_size_chunk():
    x = torch.arange(5, dtype=torch.float64)
    chunks = split_with_sizes_copy(x, [0, 5], 0)
    assert chunks[0].shape == (0,)
    torch.testing.assert_close(chunks[1], x, rtol=0, atol=0)


def test_split_with_sizes_copy_invalid_dim_asserts():
    x = torch.randn(3, 4)
    with pytest.raises(AssertionError, match="Invalid dim"):
        split_with_sizes_copy(x, [4], 2)


def test_split_with_sizes_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.split_with_sizes_copy")
    launches = []
    real_run = mod.split_indexed_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.split_indexed_copy_kernel, "run", spy_run)
    x = torch.arange(2 * 6 * 3, dtype=torch.float64).reshape(2, 6, 3)
    chunks = split_with_sizes_copy(x, [1, 4, 1], 1)
    # dim != 0 goes through the Triton gather (dim == 0 uses the clone fast path).
    assert len(launches) >= 1
    torch.testing.assert_close(chunks[0], x[:, 0:1, :], rtol=0, atol=0)
    torch.testing.assert_close(chunks[1], x[:, 1:5, :], rtol=0, atol=0)
    torch.testing.assert_close(chunks[2], x[:, 5:6, :], rtol=0, atol=0)
