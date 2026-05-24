import pytest
import torch

from .addmv import addmv, addmv_out


@pytest.mark.parametrize("N, M", [(16, 16), (32, 64), (33, 65)])
@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0), (0.5, 2.0)])
def test_addmv(N, M, alpha, beta):
    torch.manual_seed(0)
    bias = torch.randn((N,), dtype=torch.float32, device="cpu")
    mat = torch.randn(N, M, dtype=torch.float32, device="cpu")
    vec = torch.randn((M,), dtype=torch.float32, device="cpu")

    ref_out = torch.addmv(bias, mat, vec, alpha=alpha, beta=beta)
    tri_out = addmv(bias, mat, vec, alpha=alpha, beta=beta)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("N, M", [(16, 16)])
def test_addmv_out(N, M):
    torch.manual_seed(0)
    bias = torch.randn((N,), dtype=torch.float32, device="cpu")
    mat = torch.randn(N, M, dtype=torch.float32, device="cpu")
    vec = torch.randn((M,), dtype=torch.float32, device="cpu")
    out = torch.empty((N,), dtype=torch.float32, device="cpu")

    ref_out = torch.addmv(bias, mat, vec)
    addmv_out(bias, mat, vec, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)
