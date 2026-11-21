import importlib

import pytest
import torch

from .reshape_as import reshape_as


def test_reshape_as_viewable_returns_view():
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    other = torch.empty(6, 4)
    out = reshape_as(x, other)
    assert out.shape == (6, 4)
    assert out.data_ptr() == x.data_ptr()  # zero-copy view when expressible
    torch.testing.assert_close(out, x.reshape(6, 4), rtol=0, atol=0)


def test_reshape_as_strided_self_copies():
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6).t()  # (6, 4)
    other = torch.empty(24)
    out = reshape_as(x, other)
    torch.testing.assert_close(out, x.reshape(-1), rtol=0, atol=0)
    assert out.data_ptr() != x.data_ptr()


def test_reshape_as_complex_copy():
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.complex128).t()
    other = torch.empty(24)
    out = reshape_as(x, other)
    torch.testing.assert_close(out, x.reshape(-1), rtol=0, atol=0)


def test_reshape_as_numel_mismatch_raises():
    x = torch.arange(6)
    other = torch.empty(2, 3, 4)
    with pytest.raises(RuntimeError, match="is invalid for input"):
        reshape_as(x, other)


def test_reshape_as_type_errors():
    with pytest.raises(TypeError, match="for 'self'"):
        reshape_as("nope", torch.empty(3))
    with pytest.raises(TypeError, match="for 'other'"):
        reshape_as(torch.empty(3), 5)


def test_reshape_as_grad_flow():
    x = torch.randn(3, 4).t().requires_grad_(True)
    other = torch.empty(12)
    out = reshape_as(x, other)
    out.sum().backward()
    assert x.grad.shape == x.shape
    assert bool((x.grad == 1).all())


def test_reshape_as_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.reshape_as")
    launches = []
    real_run = mod._reshape_as_gather_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._reshape_as_gather_kernel, "run", spy_run)
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6).t()
    out = reshape_as(x, torch.empty(24))
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.reshape(-1), rtol=0, atol=0)
