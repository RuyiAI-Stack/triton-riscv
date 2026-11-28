import importlib

import pytest
import torch

from .split_copy import split_copy


def test_split_copy_matches_aten():
    x = torch.arange(2 * 7 * 3, dtype=torch.float64).reshape(2, 7, 3)
    for split_size, dim in [(3, 1), (7, 1), (2, 0), (1, 2), (10, 1), (3, -2)]:
        chunks = split_copy(x, split_size, dim)
        ref = torch.ops.aten.split_copy(x, split_size, dim)
        assert len(chunks) == len(ref)
        for c, r in zip(chunks, ref):
            torch.testing.assert_close(c, r, rtol=0, atol=0)


def test_split_copy_last_chunk_short():
    x = torch.arange(10, dtype=torch.float64)
    chunks = split_copy(x, 3, 0)
    shapes = [c.shape[0] for c in chunks]
    assert shapes == [3, 3, 3, 1]


def test_split_copy_zero_size_dim():
    x = torch.empty(2, 0, 3)
    chunks = split_copy(x, 0, 1)
    assert len(chunks) == 1 and chunks[0].shape == (2, 0, 3)


def test_split_copy_split_size_zero_nonempty_raises():
    x = torch.arange(5)
    with pytest.raises(RuntimeError, match="can only be 0 if dimension size is 0"):
        split_copy(x, 0, 0)


def test_split_copy_negative_split_size_raises():
    x = torch.arange(5)
    with pytest.raises(RuntimeError, match="split_size be non-negative"):
        split_copy(x, -1, 0)


def test_split_copy_zero_dim_input_raises():
    x = torch.tensor(1.0)
    with pytest.raises(RuntimeError, match="at least a 1-dimensional"):
        split_copy(x, 1, 0)


def test_split_copy_strided_input():
    x = torch.randn(4, 8).t()  # (8, 4) non-contiguous
    chunks = split_copy(x, 2, 0)
    ref = torch.ops.aten.split_copy(x, 2, 0)
    for c, r in zip(chunks, ref):
        torch.testing.assert_close(c, r, rtol=0, atol=0)


def test_split_copy_chunks_are_value_copies():
    x = torch.arange(6, dtype=torch.float32)
    chunks = split_copy(x, 3, 0)
    chunks[0][0] = -1.0
    assert x[0] == 0.0


def test_split_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.split_copy")
    launches = []
    real_run = mod._split_copy_contig_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._split_copy_contig_kernel, "run", spy_run)
    x = torch.arange(2 * 5 * 3, dtype=torch.float64).reshape(2, 5, 3)
    chunks = split_copy(x, 2, 1)
    assert len(launches) == 3, "one launch per chunk"
    for i, c in enumerate(chunks):
        torch.testing.assert_close(c, x[:, 2 * i : 2 * i + 2, :], rtol=0, atol=0)


def test_split_copy_strided_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.split_copy")
    launches = []
    real_run = mod._split_copy_strided_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._split_copy_strided_kernel, "run", spy_run)
    x = torch.randn(3, 6).t()  # (6, 3) non-contiguous
    chunks = split_copy(x, 4, 0)
    assert len(launches) == 2
    torch.testing.assert_close(chunks[0], x[:4], rtol=0, atol=0)
    torch.testing.assert_close(chunks[1], x[4:], rtol=0, atol=0)
