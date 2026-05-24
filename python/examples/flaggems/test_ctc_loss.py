import pytest
import torch
import torch.nn.functional as F

from .ctc_loss import ctc_loss


@pytest.mark.parametrize("T, N, C, S", [(10, 2, 5, 3), (20, 1, 10, 4)])
def test_ctc_loss(T, N, C, S):
    torch.manual_seed(0)
    log_probs = torch.randn(
        T, N, C, device="cpu", dtype=torch.float32
    ).log_softmax(2)
    targets = torch.randint(1, C, (N, S), device="cpu", dtype=torch.long)
    input_lengths = torch.full((N,), T, dtype=torch.long)
    target_lengths = torch.full((N,), S, dtype=torch.long)

    ref = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    tri = ctc_loss(log_probs, targets, input_lengths, target_lengths)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_ctc_loss_blank():
    torch.manual_seed(0)
    T, N, C = 10, 1, 5
    log_probs = torch.randn(
        T, N, C, device="cpu", dtype=torch.float32
    ).log_softmax(2)
    targets = torch.randint(1, C, (N, 2), device="cpu", dtype=torch.long)
    input_lengths = torch.full((N,), T, dtype=torch.long)
    target_lengths = torch.full((N,), 2, dtype=torch.long)

    ref = F.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=0
    )
    tri = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
