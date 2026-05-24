import pytest
import torch

from .fp8_mqa_logits import fp8_mqa_logits


@pytest.mark.parametrize("M, H, D, N", [(4, 2, 8, 6), (2, 1, 4, 3)])
def test_fp8_mqa_logits(M, H, D, N):
    torch.manual_seed(0)
    q = torch.randn(M, H, D, dtype=torch.float32)
    k = torch.randn(N, D, dtype=torch.float32)
    k_scales = torch.ones(N, dtype=torch.float32)
    weights = torch.randn(M, H, dtype=torch.float32)
    cu_seqlen_ks = torch.zeros(M, dtype=torch.int32)
    cu_seqlen_ke = torch.full((M,), N, dtype=torch.int32)

    tri = fp8_mqa_logits(
        q,
        (k, k_scales),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
    )

    # Reference: manual computation
    score = torch.zeros(M, N, dtype=torch.float32)
    for h in range(H):
        q_h = q[:, h, :]
        score_h = torch.mm(q_h, k.T)
        score_h = torch.relu(score_h)
        score += score_h * weights[:, h : h + 1]
    ref = score

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
