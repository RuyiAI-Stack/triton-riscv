import pytest
import torch

from .mm_streamk import mm_streamk


@pytest.mark.parametrize(
    "M, N, K",
    [(512, 512, 512), (256, 128, 512), (128, 256, 1024), (512, 512, 1023)],
)
def test_mm_streamk(M, N, K):
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)

    ref_out = torch.mm(a, b)
    tri_out = mm_streamk(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
