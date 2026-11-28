import importlib

import pytest
import torch

from .flipud import flipud


def test_flipud_matches_aten():
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    torch.testing.assert_close(flipud(x), torch.flipud(x), rtol=0, atol=0)
    x1 = torch.arange(7, dtype=torch.float64)
    torch.testing.assert_close(flipud(x1), torch.flipud(x1), rtol=0, atol=0)


def test_flipud_strided_input():
    x = torch.randn(4, 6).t()
    torch.testing.assert_close(flipud(x), torch.flipud(x), rtol=0, atol=0)


def test_flipud_complex():
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=torch.complex128)
    torch.testing.assert_close(flipud(x), torch.flipud(x), rtol=0, atol=0)


def test_flipud_grad_flow():
    x = torch.arange(6, dtype=torch.float64, requires_grad=True)
    out = flipud(x)
    out.sum().backward()
    torch.testing.assert_close(
        x.grad, torch.ones(6, dtype=torch.float64), rtol=0, atol=0
    )


def test_flipud_0d_raises():
    with pytest.raises(RuntimeError, match=">= 1-d"):
        flipud(torch.tensor(1.0))


def test_flipud_empty():
    x = torch.empty(0, 3)
    assert flipud(x).shape == (0, 3)


def test_flipud_contig_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.flipud")
    launches = []
    real_run = mod.flipud_contiguous_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.flipud_contiguous_kernel, "run", spy_run)
    x = torch.arange(2 * 5, dtype=torch.float64).reshape(2, 5)
    out = flipud(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.flip(0), rtol=0, atol=0)


def test_flipud_strided_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.flipud")
    launches = []
    real_run = mod.flipud_strided_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.flipud_strided_kernel, "run", spy_run)
    x = torch.randn(4, 6).t()
    out = flipud(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.flip(0), rtol=0, atol=0)
