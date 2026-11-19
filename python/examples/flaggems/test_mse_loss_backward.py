import importlib

import pytest
import torch

from .mse_loss_backward import mse_loss_backward


@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mse_loss_backward_matches_aten(reduction, dtype):
    torch.manual_seed(0)
    self_t = torch.randn(4, 5, dtype=dtype)
    target = torch.randn(4, 5, dtype=dtype)
    grad = (
        torch.randn(4, 5, dtype=dtype)
        if reduction == 0
        else torch.randn((), dtype=dtype)
    )

    ref = torch.ops.aten.mse_loss_backward(grad, self_t, target, reduction)
    tri = mse_loss_backward(grad, self_t, target, reduction)

    assert tri.dtype == ref.dtype and tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_mse_loss_backward_reduction_scaling():
    # Probed on torch CPU: reduction=1 divides by numel, 0 and 2 do not.
    x = torch.tensor([2.0])
    t = torch.tensor([1.0])
    g = torch.tensor(1.0)
    assert mse_loss_backward(g, x, t, 0).item() == pytest.approx(2.0)
    assert mse_loss_backward(g, x, t, 1).item() == pytest.approx(2.0)
    assert mse_loss_backward(g, x, t, 2).item() == pytest.approx(2.0)

    x2 = torch.tensor([2.0, 4.0])
    t2 = torch.tensor([1.0, 1.0])
    # mean divides by 2: elementwise 2*(x-t)/2 = (x-t).
    out = mse_loss_backward(g, x2, t2, 1)
    torch.testing.assert_close(out, torch.tensor([1.0, 3.0]), rtol=0.0, atol=1e-7)


def test_mse_loss_backward_empty_none():
    empty = torch.empty(0)
    out = mse_loss_backward(empty, empty, empty, 0)
    assert out.shape == (0,)


def test_mse_loss_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.mse_loss_backward")
    launches = []
    real_run = mod._mse_loss_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._mse_loss_backward_kernel, "run", spy_run)
    self_t = torch.randn(1024)
    target = torch.randn(1024)
    grad = torch.randn(1024)
    out = mod.mse_loss_backward(grad, self_t, target, 0)
    assert len(launches) >= 1, "mse_loss_backward must launch the Triton kernel"
    torch.testing.assert_close(
        out,
        torch.ops.aten.mse_loss_backward(grad, self_t, target, 0),
        rtol=0.0,
        atol=1e-6,
    )
