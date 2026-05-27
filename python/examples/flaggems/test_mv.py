import pytest
import torch

from .mv import mv


@pytest.mark.parametrize(
    "M, N",
    [(512, 512), (256, 1023), (128, 1024), (1024, 512)],
)
def test_mv(M, N):
    torch.manual_seed(0)
    a = torch.randn(M, N, device="cpu", dtype=torch.float32)
    v = torch.randn(N, device="cpu", dtype=torch.float32)

    ref = torch.mv(a, v)
    tri = mv(a, v)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
