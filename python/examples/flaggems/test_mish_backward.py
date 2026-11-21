import importlib

import pytest
import torch

from .mish_backward import mish_backward


def _tolerance(dtype):
    # fp16 runs the fp32 compute path (suite band); fp32 keeps the measured
    # 1e-5 softplus/tanh budget; fp64 is native double.
    if dtype == torch.float16:
        return 1e-2, 1e-3
    return 0, 1e-5


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_mish_backward_matches_aten(size, dtype):
    torch.manual_seed(0)
    grad = torch.randn(size, dtype=dtype)
    self_t = torch.randn(size, dtype=dtype) * 3.0

    ref = torch.ops.aten.mish_backward(grad, self_t)
    tri = mish_backward(grad, self_t)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_mish_backward_anchor_at_zero():
    # Probed on torch CPU: f'(0) = tanh(softplus(0)) = tanh(ln 2) = 0.6.
    g = torch.ones(1)
    x = torch.zeros(1)
    tri = mish_backward(g, x)
    assert abs(tri[0].item() - 0.6) <= 1e-5


def test_mish_backward_empty():
    empty = torch.empty(0)
    assert mish_backward(empty, empty).shape == (0,)


def test_mish_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.mish_backward")
    launches = []
    real_run = mod._mish_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._mish_backward_kernel, "run", spy_run)
    g = torch.randn(1024)
    x = torch.randn(1024) * 3.0
    out = mod.mish_backward(g, x)
    assert len(launches) >= 1, "mish_backward must launch the Triton kernel"
    torch.testing.assert_close(
        out, torch.ops.aten.mish_backward(g, x), rtol=0, atol=1e-5
    )
