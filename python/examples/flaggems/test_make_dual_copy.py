import importlib

import torch

from .make_dual_copy import _make_dual_copy, _make_dual_copy_out


def test_make_dual_copy_values_match_primal():
    for dt in (torch.float64, torch.float32, torch.int64):
        primal = (torch.arange(12) * 1.5).to(dt).reshape(3, 4)
        tangent = torch.randn(3, 4)  # accepted but contributes no values
        out = _make_dual_copy(primal, tangent, 0)
        torch.testing.assert_close(out, primal, rtol=0, atol=0)
        assert out is not primal


def test_make_dual_copy_strided_primal():
    primal = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6).t()
    tangent = torch.randn(6, 4)
    out = _make_dual_copy(primal, tangent, 0)
    torch.testing.assert_close(out, primal.contiguous(), rtol=0, atol=0)


def test_make_dual_copy_complex128_fallback():
    primal = torch.randn(3, 4, dtype=torch.complex128)
    tangent = torch.randn(3, 4, dtype=torch.complex128)
    out = _make_dual_copy(primal, tangent, 0)
    torch.testing.assert_close(out, primal, rtol=0, atol=0)


def test_make_dual_copy_empty():
    primal = torch.empty(0, 3)
    tangent = torch.empty(0, 3)
    out = _make_dual_copy(primal, tangent, 0)
    assert out.shape == (0, 3)


def test_make_dual_copy_out_variant():
    primal = torch.arange(6, dtype=torch.float64)
    out = torch.empty(6, dtype=torch.float64)
    ret = _make_dual_copy_out(primal, None, 0, out=out)
    assert ret is out
    torch.testing.assert_close(out, primal, rtol=0, atol=0)


def test_make_dual_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.make_dual_copy")
    launches = []
    real_run = mod._copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._copy_kernel, "run", spy_run)
    primal = torch.arange(24, dtype=torch.float64).reshape(2, 12)
    out = _make_dual_copy(primal, None, 0)
    assert len(launches) >= 1
    torch.testing.assert_close(out, primal, rtol=0, atol=0)
