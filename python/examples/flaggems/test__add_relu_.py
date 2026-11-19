import importlib

import torch

from ._add_relu_ import _add_relu_


def test_add_relu_inplace_mutates_and_returns_same_tensor():
    a = torch.tensor([-1.0, 0.5, 2.0])
    b = torch.tensor([1.0, -1.0, 0.5])
    original = a.clone()

    out = _add_relu_(a, b)

    assert out.data_ptr() == a.data_ptr(), "must return the same storage"
    torch.testing.assert_close(a, torch.relu(original + b), rtol=0.0, atol=0.0)


def test_add_relu_inplace_alpha():
    a = torch.tensor([-2.0, 1.0])
    b = torch.tensor([1.0, 1.0])
    out = _add_relu_(a, b, alpha=2.0)
    assert out.data_ptr() == a.data_ptr()
    torch.testing.assert_close(a, torch.tensor([0.0, 3.0]), rtol=0.0, atol=0.0)


def test_add_relu_inplace_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._add_relu_")
    launches = []
    real_run = mod._add_relu_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._add_relu_kernel, "run", spy_run)
    torch.manual_seed(0)
    a = torch.randn(1024)
    b = torch.randn(1024)
    expected = torch.relu(a.clone() + b)
    out = mod._add_relu_(a, b)
    assert len(launches) >= 1, "_add_relu_ must launch the Triton kernel"
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)


def test_add_relu_inplace_empty():
    a = torch.empty(0)
    out = _add_relu_(a, torch.empty(0))
    assert out.data_ptr() == a.data_ptr() and out.numel() == 0
