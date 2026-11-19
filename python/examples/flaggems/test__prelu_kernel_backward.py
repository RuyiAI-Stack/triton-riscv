import importlib

import pytest
import torch

from ._prelu_kernel_backward import _prelu_kernel_backward


def _autograd_grads(x, w, grad_out):
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    out = torch.nn.functional.prelu(xa, wa)
    out.backward(grad_out)
    return xa.grad, wa.grad


def test_prelu_kernel_backward_matches_autograd():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 4)
    w = torch.randn(3)
    grad = torch.randn(2, 3, 4, 4)

    ref_gi, ref_gw = _autograd_grads(x, w, grad)
    gi, gw = _prelu_kernel_backward(grad, x, w)

    assert gi.shape == ref_gi.shape
    assert gw.shape == ref_gw.shape
    torch.testing.assert_close(gi, ref_gi, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(gw, ref_gw, rtol=0.0, atol=1e-6)


def test_prelu_kernel_backward_scalar_weight():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    w = torch.tensor([-0.5, 0.25, 1.5])
    grad = torch.randn(2, 3, 8)

    ref_gi, ref_gw = _autograd_grads(x, w, grad)
    gi, gw = _prelu_kernel_backward(grad, x, w)
    torch.testing.assert_close(gi, ref_gi, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(gw, ref_gw, rtol=0.0, atol=1e-6)


def test_prelu_kernel_backward_missing_args():
    with pytest.raises(ValueError):
        _prelu_kernel_backward(torch.randn(3))


def test_prelu_kernel_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._prelu_kernel_backward")
    launches = []
    real_run = mod._prelu_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._prelu_backward_kernel, "run", spy_run)
    x = torch.randn(2, 3, 4, 4)
    w = torch.randn(3)
    grad = torch.randn(2, 3, 4, 4)
    gi, gw = mod._prelu_kernel_backward(grad, x, w)
    assert len(launches) >= 1, "_prelu_kernel_backward must launch the Triton kernel"
    ref_gi, ref_gw = _autograd_grads(x, w, grad)
    torch.testing.assert_close(gi, ref_gi, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(gw, ref_gw, rtol=0.0, atol=1e-6)


def test_prelu_kernel_backward_empty():
    x = torch.empty(0, 3, 4, 4)
    w = torch.randn(3)
    grad = torch.empty(0, 3, 4, 4)
    gi, gw = _prelu_kernel_backward(grad, x, w)
    assert gi.shape == (0, 3, 4, 4)
    assert gw.shape == (3,) and bool((gw == 0).all())
