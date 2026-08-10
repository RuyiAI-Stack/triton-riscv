import pytest
import torch

from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out


@pytest.mark.parametrize("M, N, K", [(16, 16, 16), (32, 64, 32), (33, 65, 33)])
@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0), (0.5, 2.0)])
def test_addmm(M, N, K, alpha, beta):
    torch.manual_seed(0)
    bias = torch.randn(N, dtype=torch.float32, device="cpu")
    mat1 = torch.randn(M, K, dtype=torch.float32, device="cpu")
    mat2 = torch.randn(K, N, dtype=torch.float32, device="cpu")

    ref_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    tri_out = addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("M, N, K", [(16, 16, 16)])
@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0)])
def test_addmm_out(M, N, K, alpha, beta):
    torch.manual_seed(0)
    bias = torch.randn((M, N), dtype=torch.float32, device="cpu")
    mat1 = torch.randn(M, K, dtype=torch.float32, device="cpu")
    mat2 = torch.randn(K, N, dtype=torch.float32, device="cpu")
    out = torch.empty((M, N), dtype=torch.float32, device="cpu")

    ref_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    addmm_out(bias, mat1, mat2, alpha=alpha, beta=beta, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("M, N, K", [(16, 16, 16)])
@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0)])
def test_addmm_dtype_default(M, N, K, alpha, beta):
    torch.manual_seed(0)
    bias = torch.randn(N, dtype=torch.float32, device="cpu")
    mat1 = torch.randn(M, K, dtype=torch.float32, device="cpu")
    mat2 = torch.randn(K, N, dtype=torch.float32, device="cpu")

    ref_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    tri_out = addmm_dtype(bias, mat1, mat2, torch.float32, alpha=alpha, beta=beta)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("M, N, K", [(16, 16, 16)])
@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0)])
def test_addmm_dtype_out(M, N, K, alpha, beta):
    torch.manual_seed(0)
    bias = torch.randn(N, dtype=torch.float32, device="cpu")
    mat1 = torch.randn(M, K, dtype=torch.float32, device="cpu")
    mat2 = torch.randn(K, N, dtype=torch.float32, device="cpu")
    out = torch.empty((M, N), dtype=torch.float32, device="cpu")

    ref_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    addmm_dtype_out(bias, mat1, mat2, torch.float32, alpha=alpha, beta=beta, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)
