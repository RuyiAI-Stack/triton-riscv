import pytest
import torch

from .addr import addr


@pytest.mark.parametrize("M, N", [(16, 16), (32, 64), (33, 65)])
@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0), (0.5, 2.0)])
def test_addr(M, N, alpha, beta):
    torch.manual_seed(0)
    inp = torch.randn((M, N), dtype=torch.float32, device="cpu")
    vec1 = torch.randn((M,), dtype=torch.float32, device="cpu")
    vec2 = torch.randn((N,), dtype=torch.float32, device="cpu")

    ref_out = torch.addr(inp, vec1, vec2, alpha=alpha, beta=beta)
    tri_out = addr(inp, vec1, vec2, alpha=alpha, beta=beta)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("M, N", [(16, 16)])
def test_addr_broadcast(M, N):
    torch.manual_seed(0)
    inp = torch.randn((1, N), dtype=torch.float32, device="cpu")
    vec1 = torch.randn((M,), dtype=torch.float32, device="cpu")
    vec2 = torch.randn((N,), dtype=torch.float32, device="cpu")

    ref_out = torch.addr(inp, vec1, vec2)
    tri_out = addr(inp, vec1, vec2)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("M, N", [(16, 16)])
def test_addr_scalar_broadcast(M, N):
    torch.manual_seed(0)
    inp = torch.tensor(5.0, dtype=torch.float32, device="cpu")
    vec1 = torch.randn((M,), dtype=torch.float32, device="cpu")
    vec2 = torch.randn((N,), dtype=torch.float32, device="cpu")

    ref_out = torch.addr(inp, vec1, vec2)
    tri_out = addr(inp, vec1, vec2)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
