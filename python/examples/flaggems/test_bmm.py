import pytest
import torch

from .bmm import bmm, bmm_out


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("M, N, K", [(128, 64, 64), (32, 16, 32)])
def test_bmm_forward(batch, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(batch, M, K, dtype=torch.float32, device="cpu")
    B = torch.randn(batch, K, N, dtype=torch.float32, device="cpu")
    ref = torch.bmm(A, B)
    tri = bmm(A, B)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("M, N, K", [(128, 64, 64)])
def test_bmm_out(batch, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(batch, M, K, dtype=torch.float32, device="cpu")
    B = torch.randn(batch, K, N, dtype=torch.float32, device="cpu")
    out = torch.empty(batch, M, N, dtype=torch.float32, device="cpu")
    ref = torch.bmm(A, B)
    bmm_out(A, B, out)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
