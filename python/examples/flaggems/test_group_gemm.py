import torch

from .group_gemm import group_mm


def test_group_mm_trivial():
    """Minimal compilation coverage test for group_mm kernel."""
    torch.manual_seed(0)
    M, N, K = 128, 64, 32
    A = torch.randn(M, K, device="cpu", dtype=torch.bfloat16)
    B = torch.randn(1, K, N, device="cpu", dtype=torch.bfloat16)
    offs = torch.tensor([M], dtype=torch.int32, device="cpu")
    _ = group_mm(A, B, offs)
