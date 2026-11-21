import importlib
import re

import pytest
import torch

from ._reshape_from_tensor import _reshape_from_tensor


def test_reshape_from_tensor_viewable_returns_view():
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    out = _reshape_from_tensor(x, torch.tensor([6, 4], dtype=torch.int64))
    torch.testing.assert_close(out, x.reshape(6, 4), rtol=0, atol=0)
    assert out.data_ptr() == x.data_ptr()  # view, not copy (upstream behavior)


def test_reshape_from_tensor_copy_path():
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6).t()  # (6, 4)
    out = _reshape_from_tensor(x, torch.tensor([4, 6], dtype=torch.int64))
    torch.testing.assert_close(out, x.reshape(4, 6), rtol=0, atol=0)
    assert out.data_ptr() != x.data_ptr()


def test_reshape_from_tensor_minus_one_inferred():
    x = torch.arange(24, dtype=torch.float64)
    out = _reshape_from_tensor(x, torch.tensor([2, -1], dtype=torch.int64))
    assert out.shape == (2, 12)


def test_reshape_from_tensor_two_minus_ones_raise():
    x = torch.arange(12, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="only one dimension"):
        _reshape_from_tensor(x, torch.tensor([-1, -1], dtype=torch.int64))


def test_reshape_from_tensor_numel_mismatch_raises():
    x = torch.arange(12, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="is invalid for input"):
        _reshape_from_tensor(x, torch.tensor([5, 3], dtype=torch.int64))


def test_reshape_from_tensor_shape_tensor_validation():
    x = torch.arange(6)
    with pytest.raises(RuntimeError, match="Expected a value of type 'Tensor'"):
        _reshape_from_tensor(x, [2, 3])
    with pytest.raises(RuntimeError, match="scalar type Long"):
        _reshape_from_tensor(x, torch.tensor([2, 3], dtype=torch.int32))
    with pytest.raises(RuntimeError, match=re.escape("dim() == 1")):
        _reshape_from_tensor(x, torch.tensor([[2, 3]], dtype=torch.int64))


def test_reshape_from_tensor_complex_copy_path():
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.complex128).t()
    out = _reshape_from_tensor(x, torch.tensor([24], dtype=torch.int64))
    torch.testing.assert_close(out, x.reshape(-1), rtol=0, atol=0)


def test_reshape_from_tensor_empty():
    x = torch.empty(0, 3, dtype=torch.float64)
    out = _reshape_from_tensor(x, torch.tensor([0, 3], dtype=torch.int64))
    assert out.shape == (0, 3)


def test_reshape_from_tensor_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._reshape_from_tensor")
    launches = []
    real_run = mod._rft_gather_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._rft_gather_kernel, "run", spy_run)
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6).t()
    out = _reshape_from_tensor(x, torch.tensor([2, 12], dtype=torch.int64))
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.reshape(2, 12), rtol=0, atol=0)
