import importlib

import torch

from ._prelu_kernel import _prelu_kernel


def test_prelu_kernel_channel_weight():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 4)
    w = torch.randn(3)
    ref = torch.nn.functional.prelu(x, w)
    tri = _prelu_kernel(x, w)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_prelu_kernel_scalar_weight_and_1d():
    x = torch.randn(64)
    w = torch.tensor([0.25])
    ref = torch.nn.functional.prelu(x, w)
    torch.testing.assert_close(_prelu_kernel(x, w), ref, rtol=0.0, atol=1e-6)


def test_prelu_kernel_empty():
    empty = torch.empty(0, 3, 2, 2)
    out = _prelu_kernel(empty, torch.randn(3))
    assert out.shape == empty.shape


def test_prelu_kernel_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._prelu_kernel")
    launches = []
    real_run = mod._prelu_forward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._prelu_forward_kernel, "run", spy_run)
    x = torch.randn(2, 3, 4, 4)
    w = torch.randn(3)
    out = mod._prelu_kernel(x, w)
    assert len(launches) >= 1, "_prelu_kernel must launch the Triton kernel"
    torch.testing.assert_close(
        out, torch.nn.functional.prelu(x, w), rtol=0.0, atol=1e-6
    )
