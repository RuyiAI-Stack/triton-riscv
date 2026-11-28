import importlib

import pytest
import torch

from .ravel import ravel


def test_ravel_contiguous_returns_view():
    x = torch.arange(2 * 3, dtype=torch.float64).reshape(2, 3)
    out = ravel(x)
    torch.testing.assert_close(out, x.reshape(-1), rtol=0, atol=0)
    assert out.data_ptr() == x.data_ptr()  # alias for contiguous input


def test_ravel_strided_input_copies():
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6).t()  # (6, 4)
    out = ravel(x)
    torch.testing.assert_close(out, x.reshape(-1), rtol=0, atol=0)
    assert out.data_ptr() != x.data_ptr()


def test_ravel_complex_strided():
    torch.manual_seed(0)
    x = torch.randn(3, 5, dtype=torch.complex128).t()
    out = ravel(x)
    torch.testing.assert_close(out, x.reshape(-1), rtol=0, atol=0)


def test_ravel_matches_aten():
    for x in [torch.randn(2, 3), torch.randn(4, 6).t(), torch.arange(7)]:
        torch.testing.assert_close(ravel(x), torch.ops.aten.ravel(x), rtol=0, atol=0)


def test_ravel_grad_flow():
    x = torch.randn(3, 4).t().requires_grad_(True)
    out = ravel(x)
    out.sum().backward()
    assert x.grad.shape == x.shape
    assert bool((x.grad == 1).all())


def test_ravel_non_tensor_raises():
    with pytest.raises(TypeError, match="must be a Tensor"):
        ravel([1, 2, 3])


def test_ravel_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.ravel")
    launches = []
    real_run = mod._ravel_gather_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._ravel_gather_kernel, "run", spy_run)
    x = torch.arange(4 * 6, dtype=torch.float64).reshape(4, 6).t()
    out = ravel(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.reshape(-1), rtol=0, atol=0)
