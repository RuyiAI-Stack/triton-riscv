import importlib

import pytest
import torch

from ._conj_copy import _conj_copy, _conj_copy_out


def test_conj_copy_matches_aten():
    torch.manual_seed(0)
    for cdt in (torch.complex64, torch.complex128):
        x = torch.randn(3, 5, dtype=cdt)
        out = _conj_copy(x)
        torch.testing.assert_close(out, torch.ops.aten._conj_copy(x), rtol=0, atol=0)


def test_conj_copy_non_complex_raises():
    x = torch.randn(3)
    with pytest.raises(RuntimeError, match="only supports complex"):
        _conj_copy(x)


def test_conj_copy_strided_input():
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.complex128).t()
    out = _conj_copy(x)
    torch.testing.assert_close(out, torch.ops.aten._conj_copy(x), rtol=0, atol=0)


def test_conj_copy_double_conj_resolves_bit():
    torch.manual_seed(0)
    x = torch.randn(3, 3, dtype=torch.complex64)
    cc = x.conj()
    out = _conj_copy(cc)  # conj(conj(x)) == x
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_conj_copy_empty():
    x = torch.empty(0, 3, dtype=torch.complex64)
    out = _conj_copy(x)
    assert out.shape == (0, 3)


def test_conj_copy_out_variant():
    torch.manual_seed(0)
    x = torch.randn(4, dtype=torch.complex128)
    out = torch.empty(4, dtype=torch.complex128)
    ret = _conj_copy_out(x, out=out)
    assert ret is out
    torch.testing.assert_close(out, x.conj(), rtol=0, atol=0)


def test_conj_copy_out_dtype_mismatch():
    x = torch.randn(4, dtype=torch.complex64)
    out = torch.empty(4, dtype=torch.complex128)
    with pytest.raises(RuntimeError, match="dtype"):
        _conj_copy_out(x, out=out)


def test_conj_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._conj_copy")
    launches = []
    real_run = mod._conj_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._conj_copy_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(6, dtype=torch.complex128)
    out = _conj_copy(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.conj(), rtol=0, atol=0)
    assert out.dtype == torch.complex128
