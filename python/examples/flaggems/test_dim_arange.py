import importlib

import torch

from .dim_arange import _dim_arange


def test_dim_arange_values():
    x = torch.empty(7, 3)
    out = _dim_arange(x, 0)
    torch.testing.assert_close(out, torch.arange(7), rtol=0, atol=0)
    assert out.dtype == torch.int64


def test_dim_arange_negative_dim_and_values_irrelevant():
    x = torch.randn(2, 9, 3)  # values must not matter
    out = _dim_arange(x, -2)
    torch.testing.assert_close(out, torch.arange(9), rtol=0, atol=0)


def test_dim_arange_exact_pow2_path():
    x = torch.empty(1024)
    out = _dim_arange(x, 0)
    torch.testing.assert_close(out, torch.arange(1024), rtol=0, atol=0)


def test_dim_arange_non_pow2_masked_path():
    x = torch.empty(1000)
    out = _dim_arange(x, 0)
    torch.testing.assert_close(out, torch.arange(1000), rtol=0, atol=0)


def test_dim_arange_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.dim_arange")
    launches = []
    real_run = mod._dim_arange_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._dim_arange_kernel, "run", spy_run)
    x = torch.empty(1000)
    out = _dim_arange(x, 0)
    assert len(launches) >= 1
    torch.testing.assert_close(out, torch.arange(1000), rtol=0, atol=0)


def test_dim_arange_exact_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.dim_arange")
    launches = []
    real_run = mod._dim_arange_kernel_exact.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._dim_arange_kernel_exact, "run", spy_run)
    x = torch.empty(2048)
    out = _dim_arange(x, 0)
    assert len(launches) >= 1
    torch.testing.assert_close(out, torch.arange(2048), rtol=0, atol=0)
