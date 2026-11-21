import importlib

import torch

from .clone import clone


def test_clone_matches_aten_contiguous():
    for dt in (torch.float64, torch.float32, torch.int64):
        x = torch.arange(2 * 3 * 4, dtype=dt).reshape(2, 3, 4)
        out = clone(x)
        ref = torch.ops.aten.clone(x)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_clone_preserves_strides():
    x = torch.arange(6 * 4, dtype=torch.float64).reshape(6, 4).t()  # (4, 6)
    out = clone(x)
    assert out.shape == (4, 6)
    assert out.stride() == x.stride()  # preserve_format keeps the layout
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_clone_is_value_copy():
    x = torch.arange(6, dtype=torch.float64)
    out = clone(x)
    out[0] = -1.0
    assert x[0] == 0.0
    assert out.data_ptr() != x.data_ptr()


def test_clone_complex():
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=torch.complex128)
    out = clone(x)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    assert out.dtype == torch.complex128


def test_clone_complex_conj_view():
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=torch.complex128)
    cv = x.conj()
    out = clone(cv)
    torch.testing.assert_close(out, cv, rtol=0, atol=0)  # logical values


def test_clone_contiguous_format_request():
    x = torch.arange(6 * 4, dtype=torch.float64).reshape(6, 4).t()
    out = clone(x, memory_format=torch.contiguous_format)
    assert out.is_contiguous()
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_clone_empty():
    x = torch.empty(0, 3, dtype=torch.float32)
    out = clone(x)
    assert out.shape == (0, 3)


def test_clone_grad_flows():
    x = torch.arange(4, dtype=torch.float64, requires_grad=True)
    out = clone(x)
    assert out.grad_fn is not None
    out.sum().backward()
    assert torch.equal(x.grad, torch.ones(4, dtype=torch.float64))


def test_clone_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.clone")
    launches = []
    real_run = mod._clone_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._clone_kernel, "run", spy_run)
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    out = clone(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x, rtol=0, atol=0)
