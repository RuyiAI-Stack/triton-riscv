import importlib

import pytest
import torch

from .logit_backward import logit_backward


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logit_backward_matches_aten(size, dtype):
    torch.manual_seed(0)
    grad = torch.randn(size, dtype=dtype)
    self_t = torch.rand(size, dtype=dtype) * 0.98 + 0.01

    ref = torch.ops.aten.logit_backward(grad, self_t, 1e-6)
    tri = logit_backward(grad, self_t, 1e-6)

    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_logit_backward_eps_clamp_endpoints():
    # Probed on torch CPU: x == eps still yields g / (eps * (1 - eps))
    # (the clamp endpoints are inclusive), while x == 0 falls outside the
    # clamped domain and yields exactly zero.
    g = torch.ones(4)
    x = torch.tensor([0.0, 1e-6, 0.5, 1.0])

    ref = torch.ops.aten.logit_backward(g, x, 1e-6)
    tri = logit_backward(g, x, 1e-6)

    torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0)
    assert tri[0].item() == 0.0 and tri[3].item() == 0.0
    torch.testing.assert_close(tri[1], torch.tensor(1.0 / (1e-6 * (1 - 1e-6))))
    torch.testing.assert_close(tri[2], torch.tensor(4.0), rtol=0.0, atol=1e-6)


def test_logit_backward_without_eps():
    # eps=None means no clamp: the whole open interval carries g/(x(1-x)).
    g = torch.randn(5)
    x = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    ref = torch.ops.aten.logit_backward(g, x, None)
    torch.testing.assert_close(logit_backward(g, x, None), ref, rtol=0.0, atol=1e-6)


def test_logit_backward_empty():
    empty = torch.empty(0)
    assert logit_backward(empty, empty, 1e-6).shape == (0,)


def test_logit_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.logit_backward")
    launches = []
    real_run = mod._logit_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._logit_backward_kernel, "run", spy_run)
    g = torch.randn(1024)
    x = torch.rand(1024) * 0.98 + 0.01
    out = mod.logit_backward(g, x, 1e-6)
    assert len(launches) >= 1, "logit_backward must launch the Triton kernel"
    torch.testing.assert_close(
        out, torch.ops.aten.logit_backward(g, x, 1e-6), rtol=0.0, atol=1e-6
    )
