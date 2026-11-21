import importlib

import pytest
import torch

from ._reshape_alias_copy import _reshape_alias_copy
from ._reshape_copy import _reshape_copy


def test_reshape_alias_copy_values():
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6)
    out = _reshape_alias_copy(x, (2, 12), (12, 1))
    torch.testing.assert_close(out, x.reshape(2, 12), rtol=0, atol=0)
    assert out.is_contiguous()


def test_reshape_alias_copy_storage_offset_reused():
    base = torch.arange(24, dtype=torch.float64)
    x = base[4:]  # storage offset 4
    out = _reshape_alias_copy(x, (2, 5), (5, 1))
    torch.testing.assert_close(out, base[4:14].reshape(2, 5), rtol=0, atol=0)


def test_reshape_alias_copy_strided_view():
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6)
    out = _reshape_alias_copy(x, (6, 4), (1, 6))  # transpose-like gather
    torch.testing.assert_close(out, x.t().contiguous(), rtol=0, atol=0)


def test_reshape_alias_copy_unvalidated_size():
    # size/stride are applied verbatim; no numel validation (upstream/aten).
    x = torch.arange(12, dtype=torch.float64)
    out = _reshape_alias_copy(x, (2, 3), (3, 1))
    assert out.shape == (2, 3)
    torch.testing.assert_close(out, x[:6].reshape(2, 3), rtol=0, atol=0)


def test_reshape_alias_copy_empty():
    x = torch.empty(0, dtype=torch.float64)
    out = _reshape_alias_copy(x, (0, 3), (3, 1))
    assert out.shape == (0, 3)


def test_reshape_alias_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._reshape_alias_copy")
    launches = []
    real_run = mod._reshape_alias_copy_strided_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._reshape_alias_copy_strided_kernel, "run", spy_run)
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6)
    out = _reshape_alias_copy(x, (6, 4), (1, 6))
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.t().contiguous(), rtol=0, atol=0)


def test_reshape_copy_matches_view_semantics():
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6)
    for size in [(24,), (2, 12), (2, 3, 4), (12, 2)]:
        out = _reshape_copy(x, size)
        torch.testing.assert_close(out, x.reshape(size), rtol=0, atol=0, msg=str(size))


def test_reshape_copy_minus_one_inferred():
    x = torch.arange(24, dtype=torch.float64)
    out = _reshape_copy(x, (2, -1, 4))
    assert out.shape == (2, 3, 4)


def test_reshape_copy_two_minus_ones_raise():
    x = torch.arange(12, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="only one dimension"):
        _reshape_copy(x, (-1, -1))


def test_reshape_copy_numel_mismatch_raises():
    x = torch.arange(12, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="is invalid for input"):
        _reshape_copy(x, (5, 3))


def test_reshape_copy_strided_input():
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6).t()  # (6, 4)
    out = _reshape_copy(x, (24,))
    torch.testing.assert_close(out, x.reshape(-1), rtol=0, atol=0)


def test_reshape_copy_empty():
    x = torch.empty(0, 3)
    out = _reshape_copy(x, (0, 3))
    assert out.shape == (0, 3)
