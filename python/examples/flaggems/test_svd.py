import pytest
import torch

from .svd import svd


@pytest.mark.parametrize("M, N", [(4, 4), (2, 8)])
def test_svd(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float32)
    ref_u, ref_s, ref_v = torch.svd(x, some=True)
    tri_u, tri_s, tri_v = svd(x, some=True)

    torch.testing.assert_close(tri_s, ref_s, rtol=1e-3, atol=1e-3)
