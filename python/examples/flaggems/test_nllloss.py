import pytest
import torch
import torch.nn.functional as F

from .nllloss import (
    nll_loss2d_backward,
    nll_loss2d_forward,
    nll_loss_backward,
    nll_loss_forward,
    nllloss,
)


@pytest.mark.parametrize("shape", [(10, 5), (32, 10)])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("ignore_index", [-100, 2])
@pytest.mark.parametrize("use_weight", [False, True])
def test_nllloss_2d(shape, reduction, ignore_index, use_weight):
    N, C = shape
    inp = F.log_softmax(torch.randn(shape, dtype=torch.float32), dim=1)
    tgt = torch.randint(0, C, (N,), dtype=torch.int64)
    tgt[0] = ignore_index
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
    tri_out = nllloss(inp, tgt, weight, reduction, ignore_index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(2, 5, 4, 4), (4, 10, 8, 8)])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("ignore_index", [-100, 2])
@pytest.mark.parametrize("use_weight", [False, True])
def test_nllloss_4d(shape, reduction, ignore_index, use_weight):
    N, C, H, W = shape
    inp = F.log_softmax(torch.randn(shape, dtype=torch.float32), dim=1)
    tgt = torch.randint(0, C, (N, H, W), dtype=torch.int64)
    tgt.reshape(-1)[0] = ignore_index
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
    tri_out = nllloss(inp, tgt, weight, reduction, ignore_index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(10, 5), (32, 10)])
@pytest.mark.parametrize("ignore_index", [-100, 2])
@pytest.mark.parametrize("use_weight", [False, True])
def test_nllloss_backward(shape, ignore_index, use_weight):
    N, C = shape
    inp = F.log_softmax(
        torch.randn(shape, dtype=torch.float32, requires_grad=True), dim=1
    )
    inp = inp.detach().requires_grad_(True)
    tgt = torch.randint(0, C, (N,), dtype=torch.int64)
    tgt[0] = ignore_index
    if use_weight:
        weight = torch.rand(C, dtype=torch.float32)
    else:
        weight = None

    ref_out = F.nll_loss(
        inp, tgt, weight=weight, reduction="mean", ignore_index=ignore_index
    )
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    ref_grad = inp.grad.clone()

    inp.grad = torch.zeros_like(inp)
    _, total_weight = nll_loss_forward(
        inp, tgt, weight=weight, reduction=1, ignore_index=ignore_index
    )
    tri_grad = nll_loss_backward(
        grad_out, inp, tgt, weight, 1, ignore_index, total_weight
    )

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(10, 5)])
def test_nll_loss_forward(shape):
    torch.manual_seed(0)
    N, C = shape
    inp = F.log_softmax(torch.randn(shape, dtype=torch.float32), dim=1)
    tgt = torch.randint(0, C, (N,), dtype=torch.int64)
    tgt[0] = -100

    ref = F.nll_loss(inp, tgt, reduction="mean", weight=None, ignore_index=-100)
    tri, _ = nll_loss_forward(inp, tgt, weight=None, reduction=1, ignore_index=-100)
    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(2, 5, 4, 4)])
def test_nll_loss2d_forward(shape):
    torch.manual_seed(0)
    N, C, H, W = shape
    inp = F.log_softmax(torch.randn(shape, dtype=torch.float32), dim=1)
    tgt = torch.randint(0, C, (N, H, W), dtype=torch.int64)
    tgt.reshape(-1)[0] = -100

    ref = F.nll_loss(inp, tgt, reduction="mean", ignore_index=-100)
    tri, _ = nll_loss2d_forward(inp, tgt, weight=None, reduction=1, ignore_index=-100)
    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(2, 5, 4, 4)])
@pytest.mark.parametrize("ignore_index", [-100, 2])
@pytest.mark.parametrize("use_weight", [False, True])
def test_nll_loss2d_backward(shape, ignore_index, use_weight):
    torch.manual_seed(0)
    N, C, H, W = shape
    inp = torch.randn(shape, dtype=torch.float32)
    log_probs = F.log_softmax(inp, dim=1).detach().requires_grad_(True)
    tgt = torch.randint(0, C, (N, H, W), dtype=torch.int64)
    tgt.reshape(-1)[0] = ignore_index
    if use_weight:
        weight = torch.rand(C, dtype=torch.float32)
    else:
        weight = None

    # Reference backward via torch
    ref = F.nll_loss(
        log_probs,
        tgt,
        weight=weight,
        reduction="mean",
        ignore_index=ignore_index,
    )
    grad_out = torch.randn_like(ref)
    ref.backward(grad_out)
    ref_dx = log_probs.grad.clone()

    # Triton backward
    log_probs.grad = None
    log_probs2 = log_probs.detach().clone().requires_grad_(True)
    _, total_weight = nll_loss2d_forward(
        log_probs2,
        tgt,
        weight=weight,
        reduction=1,
        ignore_index=ignore_index,
    )
    tri_dx = nll_loss2d_backward(
        grad_out,
        log_probs2,
        tgt,
        weight=weight,
        reduction=1,
        ignore_index=ignore_index,
        total_weight=total_weight,
    )

    torch.testing.assert_close(tri_dx, ref_dx, rtol=1e-3, atol=1e-3)
