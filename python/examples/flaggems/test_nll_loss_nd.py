import pytest
import torch
import torch.nn.functional as F

from .nll_loss_nd import nll_loss_nd, nll_loss_nd_backward, nll_loss_nd_forward


@pytest.mark.parametrize("shape", [(2, 5, 4, 4), (4, 10, 8, 8), (2, 5, 4, 4, 4)])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("ignore_index", [-100, 2])
@pytest.mark.parametrize("use_weight", [False, True])
def test_nll_loss_nd(shape, reduction, ignore_index, use_weight):
    C = shape[1]
    inp = F.log_softmax(
        torch.randn(shape, dtype=torch.float32, requires_grad=True), dim=1
    )
    inp = inp.detach().requires_grad_(True)

    tgt_shape = shape[:1] + shape[2:]
    tgt = torch.randint(0, C, tgt_shape, dtype=torch.int64)
    if use_weight:
        weight = torch.rand(C, dtype=torch.float32)
    else:
        weight = None

    ref_out = F.nll_loss(
        inp,
        tgt,
        weight=weight,
        reduction={0: "none", 1: "mean", 2: "sum"}[reduction],
        ignore_index=ignore_index,
    )
    tri_out = nll_loss_nd(inp, tgt, weight, reduction, ignore_index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)

    if reduction != 0:
        grad_out = torch.randn_like(ref_out)
    else:
        grad_out = torch.randn(tgt_shape, dtype=torch.float32)

    ref_out.backward(grad_out, retain_graph=True)
    ref_grad = inp.grad.clone()
    inp.grad.zero_()

    tri_out.backward(grad_out, retain_graph=True)
    tri_grad = inp.grad.clone()

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(2, 4, 8)])
def test_nll_loss_nd_forward_backward(shape):
    torch.manual_seed(0)
    N, C, S = shape
    log_probs = (
        torch.nn.functional.log_softmax(
            torch.randn(N, C, S, dtype=torch.float32, device="cpu"), dim=1
        )
        .detach()
        .requires_grad_(True)
    )
    target = torch.randint(0, C, (N, S), device="cpu", dtype=torch.long)

    # Reference
    ref = F.nll_loss(log_probs, target, reduction="mean")
    grad_out = torch.randn_like(ref)
    ref.backward(grad_out)
    ref_dx = log_probs.grad.clone()

    # Triton forward/backward functions directly
    log_probs2 = log_probs.detach().clone().requires_grad_(True)
    tri_out, total_weight = nll_loss_nd_forward(log_probs2, target, reduction=1)
    torch.testing.assert_close(tri_out, ref, rtol=1e-3, atol=1e-3)

    tri_dx = nll_loss_nd_backward(
        grad_out,
        log_probs2,
        target,
        reduction=1,
        total_weight=total_weight,
    )
    torch.testing.assert_close(tri_dx, ref_dx, rtol=1e-3, atol=1e-3)
