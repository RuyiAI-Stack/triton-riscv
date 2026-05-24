import pytest
import torch

from .logsumexp import logsumexp


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logsumexp_1d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.logsumexp(x, dim=0)
    tri_out = logsumexp(x, dim=0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape, dim", [((4, 128), 1), ((16, 64), 0)])
def test_logsumexp_2d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.logsumexp(x, dim=dim)
    tri_out = logsumexp(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape, dim", [((4, 128), 1), ((16, 64), 0)])
def test_logsumexp_keepdim(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.logsumexp(x, dim=dim, keepdim=True)
    tri_out = logsumexp(x, dim=dim, keepdim=True)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
