import pytest
import torch

from .svd import svd


def _torch_has_lapack() -> bool:
    try:
        torch.linalg.svd(torch.eye(2))
        return True
    except RuntimeError as exc:
        if "LAPACK" in str(exc):
            return False
        raise


@pytest.mark.skipif(
    not _torch_has_lapack(),
    reason="PyTorch was built without LAPACK; torch.svd is unavailable",
)
@pytest.mark.parametrize("M, N", [(4, 4), (2, 8)])
def test_svd(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float32)
    ref_u, ref_s, ref_v = torch.svd(x, some=True)
    tri_u, tri_s, tri_v = svd(x, some=True)

    torch.testing.assert_close(tri_s, ref_s, rtol=1e-3, atol=1e-3)
