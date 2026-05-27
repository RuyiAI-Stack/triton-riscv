import pytest
import torch

from .baddbmm import baddbmm, baddbmm_out


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize(
    "M, N, K", [(128, 128, 64), (32, 16, 32), (64, 128, 32)]
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_baddbmm_forward(batch, M, N, K, dtype):
    torch.manual_seed(0)
    A = torch.randn(batch, M, K, dtype=dtype, device="cpu")
    B = torch.randn(batch, K, N, dtype=dtype, device="cpu")
    bias = torch.randn(batch, M, N, dtype=dtype, device="cpu")
    alpha = 1.0
    beta = 1.0

    ref_out = torch.baddbmm(bias, A, B, beta=beta, alpha=alpha)
    tri_out = baddbmm(bias, A, B, beta=beta, alpha=alpha)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("M, N, K", [(128, 128, 64), (32, 16, 32)])
def test_baddbmm_alpha_beta(batch, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(batch, M, K, dtype=torch.float32, device="cpu")
    B = torch.randn(batch, K, N, dtype=torch.float32, device="cpu")
    bias = torch.randn(batch, M, N, dtype=torch.float32, device="cpu")

    ref_out = torch.baddbmm(bias, A, B, beta=0.5, alpha=2.0)
    tri_out = baddbmm(bias, A, B, beta=0.5, alpha=2.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("M, N, K", [(128, 128, 64)])
def test_baddbmm_out(batch, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(batch, M, K, dtype=torch.float32, device="cpu")
    B = torch.randn(batch, K, N, dtype=torch.float32, device="cpu")
    bias = torch.randn(batch, M, N, dtype=torch.float32, device="cpu")
    out = torch.empty(batch, M, N, dtype=torch.float32, device="cpu")

    ref_out = torch.baddbmm(bias, A, B, beta=1.0, alpha=1.0)
    baddbmm_out(bias, A, B, beta=1.0, alpha=1.0, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("M, N, K", [(128, 128, 64)])
def test_baddbmm_autograd(batch, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(
        batch, M, K, dtype=torch.float32, device="cpu", requires_grad=True
    )
    B = torch.randn(
        batch, K, N, dtype=torch.float32, device="cpu", requires_grad=True
    )
    bias = torch.randn(
        batch, M, N, dtype=torch.float32, device="cpu", requires_grad=True
    )

    dgrad = torch.randn(batch, M, N, dtype=torch.float32, device="cpu")

    ref_out = torch.baddbmm(bias, A, B, beta=1.0, alpha=1.0)
    tri_out = baddbmm(bias, A, B, beta=1.0, alpha=1.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)

    ref_out.backward(dgrad)
    ref_dA = A.grad.clone()
    ref_dB = B.grad.clone()
    ref_dbias = bias.grad.clone()

    A.grad = None
    B.grad = None
    bias.grad = None
    tri_out.backward(dgrad)
    tri_dA, tri_dB, tri_dbias = A.grad, B.grad, bias.grad

    torch.testing.assert_close(tri_dA, ref_dA, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_dB, ref_dB, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_dbias, ref_dbias, rtol=1e-4, atol=1e-4)
