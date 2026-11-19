import importlib

import torch

from ._ctc_loss_backward import _ctc_loss_backward


def _case(T=10, N=2, C=6, S=4, seed=0):
    torch.manual_seed(seed)
    lp = torch.randn(T, N, C).log_softmax(2)
    labels = [c for c in range(C)]
    tg = torch.tensor(labels)[torch.randint(1, C, (N, S))]
    il = torch.full((N,), T, dtype=torch.long)
    tl = torch.randint(1, S + 1, (N,))
    return lp, tg, il, tl


def test_ctc_loss_backward_matches_autograd():
    for seed in range(3):
        lp, tg, il, tl = _case(seed=seed)
        lp_leaf = lp.detach().clone().requires_grad_(True)
        loss = torch.nn.functional.ctc_loss(lp_leaf, tg, il, tl, reduction="none")
        loss.backward(torch.ones_like(loss))
        ref = lp_leaf.grad

        nll, alpha = torch.ops.aten._ctc_loss(
            lp, tg, il.tolist(), tl.tolist(), 0, False
        )
        tri = _ctc_loss_backward(
            torch.ones(lp.shape[1]), lp, tg, il.tolist(), tl.tolist(), nll, alpha, 0
        )
        assert tri.shape == ref.shape
        assert bool(torch.isfinite(tri).all())
        torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-4)


def test_ctc_loss_backward_respects_upstream_grad():
    lp, tg, il, tl = _case()
    lp_leaf = lp.detach().clone().requires_grad_(True)
    loss = torch.nn.functional.ctc_loss(lp_leaf, tg, il, tl, reduction="none")
    w = torch.tensor([0.5, 2.0])
    loss.backward(w)
    ref = lp_leaf.grad

    nll, alpha = torch.ops.aten._ctc_loss(lp, tg, il.tolist(), tl.tolist(), 0, False)
    tri = _ctc_loss_backward(w, lp, tg, il.tolist(), tl.tolist(), nll, alpha, 0)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-4)


def test_ctc_loss_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._ctc_loss_backward")
    launches = []
    real_run = mod._ctc_loss_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._ctc_loss_backward_kernel, "run", spy_run)
    lp, tg, il, tl = _case(T=6)
    nll, alpha = torch.ops.aten._ctc_loss(lp, tg, il.tolist(), tl.tolist(), 0, False)
    out = mod._ctc_loss_backward(
        torch.ones(2), lp, tg, il.tolist(), tl.tolist(), nll, alpha, 0
    )
    assert len(launches) >= 1, "_ctc_loss_backward must launch the Triton kernel"
    lp_leaf = lp.detach().clone().requires_grad_(True)
    loss = torch.nn.functional.ctc_loss(lp_leaf, tg, il, tl, reduction="none")
    loss.backward(torch.ones_like(loss))
    torch.testing.assert_close(out, lp_leaf.grad, rtol=0.0, atol=1e-4)


def test_ctc_loss_backward_short_input_lengths():
    # beta seeds at t = il - 1 per sample; exercise samples with il < T.
    torch.manual_seed(5)
    T, N, C, S = 12, 2, 6, 4
    lp = torch.randn(T, N, C).log_softmax(2)
    tg = torch.randint(1, C, (N, S))
    il = torch.tensor([12, 7])
    tl = torch.tensor([3, 4])

    lp_leaf = lp.detach().clone().requires_grad_(True)
    loss = torch.nn.functional.ctc_loss(lp_leaf, tg, il, tl, reduction="none")
    loss.backward(torch.ones_like(loss))
    ref = lp_leaf.grad

    nll, alpha = torch.ops.aten._ctc_loss(lp, tg, il.tolist(), tl.tolist(), 0, False)
    tri = _ctc_loss_backward(
        torch.ones(N), lp, tg, il.tolist(), tl.tolist(), nll, alpha, 0
    )
    assert bool(torch.isfinite(tri).all())
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-4)
