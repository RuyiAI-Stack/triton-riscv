import pytest
import torch

from .mm import mm, mm_out


@pytest.mark.parametrize(
    "M, N, K",
    [(512, 512, 512), (256, 128, 512), (128, 256, 1024), (512, 512, 1023)],
)
def test_mm(M, N, K):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cpu", dtype=torch.float32)
    b = torch.randn(K, N, device="cpu", dtype=torch.float32)

    ref_out = torch.mm(a, b)
    tri_out = mm(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M, N, K", [(256, 256, 256), (128, 64, 256)])
def test_mm_out(M, N, K):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cpu", dtype=torch.float32)
    b = torch.randn(K, N, device="cpu", dtype=torch.float32)
    out = torch.empty((M, N), device="cpu", dtype=torch.float32)

    ref_out = torch.mm(a, b)
    mm_out(a, b, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
