import importlib

import pytest
import torch

from ._rowwise_prune import _rowwise_prune


def test_rowwise_prune_basic():
    w = torch.arange(5 * 3, dtype=torch.float64).reshape(5, 3)
    mask = torch.tensor([True, False, True, True, False])
    out, idx = _rowwise_prune(w, mask, torch.int64)
    assert out.shape == (3, 3)
    torch.testing.assert_close(out, w[[0, 2, 3]], rtol=0, atol=0)
    torch.testing.assert_close(
        idx, torch.tensor([0, -1, 1, 2, -1], dtype=torch.int64), rtol=0, atol=0
    )


def test_rowwise_prune_int32_indices():
    w = torch.randn(4, 6)
    mask = torch.tensor([False, True, True, False])
    out, idx = _rowwise_prune(w, mask, torch.int32)
    assert idx.dtype == torch.int32
    torch.testing.assert_close(out, w[[1, 2]], rtol=0, atol=0)


def test_rowwise_prune_all_kept_and_none_kept():
    w = torch.arange(3 * 2, dtype=torch.float64).reshape(3, 2)
    out_all, idx_all = _rowwise_prune(w, torch.ones(3, dtype=torch.bool), torch.int64)
    torch.testing.assert_close(out_all, w, rtol=0, atol=0)
    out_none, idx_none = _rowwise_prune(
        w, torch.zeros(3, dtype=torch.bool), torch.int64
    )
    assert out_none.shape == (0, 2)


def test_rowwise_prune_multi_block_indices():
    # M > 1024 exercises the count/prefix/multi three-kernel path.
    M, K = 3000, 5
    w = torch.arange(M * K, dtype=torch.float64).reshape(M, K)
    g = torch.Generator().manual_seed(7)
    mask = torch.rand(M, generator=g) > 0.5
    out, idx = _rowwise_prune(w, mask, torch.int64)
    keep = torch.nonzero(mask).reshape(-1)
    assert out.shape == (keep.numel(), K)
    torch.testing.assert_close(out, w[keep], rtol=0, atol=0)
    expect = torch.full((M,), -1, dtype=torch.int64)
    expect[keep] = torch.arange(keep.numel())
    torch.testing.assert_close(idx, expect, rtol=0, atol=0)


def test_rowwise_prune_errors():
    w3 = torch.randn(2, 3, 4)
    with pytest.raises(RuntimeError, match="should have 2 dimensions"):
        _rowwise_prune(w3, torch.ones(2, dtype=torch.bool), torch.int64)
    w = torch.randn(4, 3)
    with pytest.raises(RuntimeError, match="Bool"):
        _rowwise_prune(w, torch.ones(4, dtype=torch.int64), torch.int64)
    with pytest.raises(RuntimeError, match="equivalent to the number of rows"):
        _rowwise_prune(w, torch.ones(3, dtype=torch.bool), torch.int64)
    with pytest.raises(RuntimeError, match="int32"):
        _rowwise_prune(w, torch.ones(4, dtype=torch.bool), torch.float32)


def test_rowwise_prune_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._rowwise_prune")
    launches = []
    real_run = mod._rowwise_prune_scatter_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._rowwise_prune_scatter_kernel, "run", spy_run)
    w = torch.arange(4 * 3, dtype=torch.float64).reshape(4, 3)
    out, _ = _rowwise_prune(w, torch.tensor([True, True, False, True]), torch.int64)
    assert len(launches) >= 1
    torch.testing.assert_close(out, w[[0, 1, 3]], rtol=0, atol=0)
