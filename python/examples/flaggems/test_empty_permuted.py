import pytest
import torch

from .empty_permuted import empty_permuted


def test_empty_permuted_strides_follow_layout():
    t = empty_permuted((2, 3, 4), (2, 1, 0))
    assert t.shape == (2, 3, 4)
    # physical order (2,1,0): dim0 innermost -> stride 1, dim1 -> 2, dim2 -> 6
    assert t.stride() == (1, 2, 6)


def test_empty_permuted_layout_identity_is_contiguous():
    t = empty_permuted((2, 3), (0, 1))
    assert t.stride() == (3, 1)


def test_empty_permuted_validation():
    with pytest.raises(RuntimeError, match="same length"):
        empty_permuted((2, 3), (0,))
    with pytest.raises(RuntimeError, match="permutation"):
        empty_permuted((2, 2), (0, 0))


def test_empty_permuted_metadata_only():
    # The storage is deliberately uninitialized; pin dtype/shape only.
    t = empty_permuted((4, 5), (1, 0), dtype=torch.int64)
    assert t.dtype == torch.int64
    assert t.shape == (4, 5)
    assert t.stride() == (1, 4)


def test_empty_permuted_triton_path(monkeypatch):
    import importlib

    mod = importlib.import_module("flaggems.empty_permuted")
    launches = []
    real_run = mod.empty_permuted_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.empty_permuted_kernel, "run", spy_run)
    t = empty_permuted((4, 5), (1, 0), dtype=torch.float64)
    assert len(launches) >= 1, "empty_permuted must touch the storage via its kernel"
    assert t.shape == (4, 5)
    assert t.stride() == (1, 4)
