import importlib

import pytest
import torch

from .multilabel_margin_loss_backward import multilabel_margin_loss_backward


def _autograd_grad(x, t, reduction, grad_out):
    xa = x.clone().requires_grad_(True)
    loss = torch.nn.functional.multilabel_margin_loss(
        xa, t, reduction={0: "none", 1: "mean", 2: "sum"}[reduction]
    )
    loss.backward(grad_out)
    return xa.grad


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multilabel_margin_loss_backward_matches_autograd(reduction):
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    t = torch.tensor(
        [[1, 3, -1, -1, -1], [0, 2, 1, -1, -1], [2, -1, -1, -1, -1], [0, 4, 2, -1, -1]]
    )
    grad_out = (
        torch.ones_like(
            torch.nn.functional.multilabel_margin_loss(x, t, reduction="none")
        )
        if reduction == 0
        else torch.ones(())
    )

    ref = _autograd_grad(x, t, reduction, grad_out)
    # aten's backward consumes the class-indexed membership mask that its
    # forward produces (labels light up columns, not sequence positions).
    is_target = torch.ops.aten.multilabel_margin_loss_forward(x, t, reduction)[1]
    tri = multilabel_margin_loss_backward(grad_out, x, t, reduction, is_target)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_multilabel_margin_loss_backward_rejects_bad_reduction():
    x = torch.randn(3, 4)
    t = torch.tensor([[1, -1, -1, -1]] * 3)
    ist = torch.ops.aten.multilabel_margin_loss_forward(x, t, 1)[1]
    with pytest.raises(RuntimeError):
        multilabel_margin_loss_backward(torch.ones(()), x, t, True, ist)
    with pytest.raises(RuntimeError):
        multilabel_margin_loss_backward(torch.ones(()), x, t, "mean", ist)


def test_multilabel_margin_loss_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.multilabel_margin_loss_backward")
    launches = []
    real_run = mod._multilabel_margin_loss_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._multilabel_margin_loss_backward_kernel, "run", spy_run)
    x = torch.randn(4, 5)
    t = torch.tensor(
        [[1, 3, -1, -1, -1], [0, 2, 1, -1, -1], [2, -1, -1, -1, -1], [0, 4, 2, -1, -1]]
    )
    is_target = torch.ops.aten.multilabel_margin_loss_forward(x, t, 1)[1]
    grad_out = torch.ones(())
    out = mod.multilabel_margin_loss_backward(grad_out, x, t, 1, is_target)
    assert len(launches) >= 1, "multilabel_margin_loss_backward must launch the kernel"
    ref = _autograd_grad(x, t, 1, torch.ones(()))
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)


def test_multilabel_margin_loss_backward_ignores_padding_tail():
    torch.manual_seed(0)
    x = torch.randn(3, 5)
    clean = torch.tensor([[1, 3, -1, -1, -1], [0, 2, 1, -1, -1], [2, -1, -1, -1, -1]])
    dirty = torch.tensor([[1, 3, -1, 4, 2], [0, 2, 1, -1, 3], [2, -1, 4, 0, 1]])

    ist_clean = torch.ops.aten.multilabel_margin_loss_forward(x, clean, 1)[1]
    ist_dirty = torch.ops.aten.multilabel_margin_loss_forward(x, dirty, 1)[1]
    g = torch.ones(())
    a = multilabel_margin_loss_backward(g, x, clean, 1, ist_clean)
    b = multilabel_margin_loss_backward(g, x, dirty, 1, ist_dirty)
    torch.testing.assert_close(a, b, rtol=0.0, atol=1e-6)


def test_multilabel_margin_loss_backward_rows_sum_to_zero():
    # Every active hinge pair contributes -1 to its positive side and +1 to
    # its negative side; with unit upstream gradients the class-wise sum of
    # a row must therefore be zero under reduction=none.
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    t = torch.tensor(
        [[1, 3, -1, -1, -1], [0, 2, 1, -1, -1], [2, -1, -1, -1, -1], [0, 4, 2, -1, -1]]
    )
    ist = torch.ops.aten.multilabel_margin_loss_forward(x, t, 0)[1]
    grad_out = torch.ones(4)
    out = multilabel_margin_loss_backward(grad_out, x, t, 0, ist)
    row_sums = out.sum(dim=1)
    torch.testing.assert_close(row_sums, torch.zeros(4), rtol=0.0, atol=1e-6)


def test_multilabel_margin_loss_backward_empty_batch():
    x = torch.randn(0, 5)
    t = torch.empty(0, 5, dtype=torch.long)
    ist = torch.ops.aten.multilabel_margin_loss_forward(x, t, 1)[1]
    out = multilabel_margin_loss_backward(torch.ones(()), x, t, 1, ist)
    assert out.shape == (0, 5)
