import importlib

import pytest
import torch

from ._stack import _stack


def test_stack_matches_aten_basic():
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = a * 10
    for dim in (-3, -1, 0, 1, 2):
        out = _stack([a, b], dim)
        ref = torch.ops.aten._stack([a, b], dim)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"dim={dim}")


def test_stack_more_than_four_uses_batches():
    ts = [torch.full((3, 4), float(i)) for i in range(9)]
    out = _stack(ts, 0)
    ref = torch.ops.aten._stack(ts, 0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (9, 3, 4)


def test_stack_dtype_promotion():
    a = torch.arange(4, dtype=torch.int32)
    b = torch.arange(4, dtype=torch.float64)
    out = _stack([a, b], 0)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, torch.ops.aten._stack([a, b], 0), rtol=0, atol=0)


def test_stack_complex():
    torch.manual_seed(0)
    a = torch.randn(2, 3, dtype=torch.complex128)
    b = torch.randn(2, 3, dtype=torch.complex128)
    out = _stack([a, b], 0)
    torch.testing.assert_close(out, torch.ops.aten._stack([a, b], 0), rtol=0, atol=0)
    assert out.dtype == torch.complex128


def test_stack_shape_mismatch_raises():
    a = torch.randn(2, 3)
    b = torch.randn(3, 2)
    with pytest.raises(RuntimeError, match="equal size"):
        _stack([a, b], 0)


def test_stack_empty_list_raises():
    with pytest.raises(RuntimeError, match="non-empty"):
        _stack([], 0)


def test_stack_dim_out_of_range():
    a = torch.randn(2, 3)
    with pytest.raises(IndexError, match="Dimension out of range"):
        _stack([a], 3)


def test_stack_zero_size_input():
    a = torch.empty(3, 0)
    out = _stack([a, a], 0)
    assert out.shape == (2, 3, 0)


def test_stack_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._stack")
    launches = []
    real_run = mod.stack_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.stack_copy_kernel, "run", spy_run)
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    out = _stack([a, a * 10, a * 100], 1)
    assert len(launches) == 3  # one launch per tensor
    ref = torch.stack([a, a * 10, a * 100], 1)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (2, 3, 3)
