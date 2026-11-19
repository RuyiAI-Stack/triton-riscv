import importlib

import pytest
import torch

from .soft_margin_loss_backward import soft_margin_loss_backward


def _tolerance(dtype):
    # fp32 keeps 1e-5 for the riscv64 aten-vs-libdevice exp headroom measured
    # on earlier ports; fp16 runs the fp32 compute path (suite band); fp64 is
    # native double.
    if dtype == torch.float16:
        return 1e-2, 1e-3
    if dtype == torch.float32:
        return 0, 1e-5
    return 0, 1e-6


@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_soft_margin_loss_backward_matches_aten(reduction, dtype):
    torch.manual_seed(0)
    self_t = torch.randn(4, 7, dtype=dtype)
    target = torch.sign(torch.randn(4, 7, dtype=dtype))
    if reduction == 0:
        grad = torch.randn(4, 7, dtype=dtype)
    else:
        grad = torch.randn((), dtype=dtype)

    ref = torch.ops.aten.soft_margin_loss_backward(grad, self_t, target, reduction)
    tri = soft_margin_loss_backward(grad, self_t, target, reduction)

    assert tri.dtype == ref.dtype and tri.shape == ref.shape
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_soft_margin_loss_backward_sign_symmetry():
    # The loss log(1 + exp(-x*t)) has dL/dx = -t * sigmoid(-x*t); flipping
    # BOTH x and t leaves the loss unchanged and flips the gradient sign
    # (both sides computed by this kernel, so the pin stays exact).
    x = torch.tensor([0.5, -1.5, 2.0])
    g = torch.ones(3)
    pos = soft_margin_loss_backward(g, x, torch.ones(3), 0)
    neg = soft_margin_loss_backward(g, -x, -torch.ones(3), 0)
    torch.testing.assert_close(pos, -neg, rtol=0.0, atol=1e-6)


def test_soft_margin_loss_backward_empty_none():
    empty = torch.empty(0)
    out = soft_margin_loss_backward(empty, empty, empty, 0)
    assert out.shape == (0,)


def test_soft_margin_loss_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.soft_margin_loss_backward")
    launches = []
    real_run = mod._soft_margin_loss_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._soft_margin_loss_backward_kernel, "run", spy_run)
    self_t = torch.randn(32, 16)
    target = torch.sign(torch.randn(32, 16))
    grad = torch.randn((), dtype=torch.float32)
    out = mod.soft_margin_loss_backward(grad, self_t, target, 1)
    assert len(launches) >= 1, "soft_margin_loss_backward must launch the kernel"
    torch.testing.assert_close(
        out,
        torch.ops.aten.soft_margin_loss_backward(grad, self_t, target, 1),
        rtol=0.0,
        atol=1e-5,
    )
