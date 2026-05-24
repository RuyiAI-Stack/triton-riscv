import pytest
import torch
import torch.nn.functional as F

from .softplus import softplus


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("beta,threshold", [(1.0, 20.0), (0.5, 10.0)])
def test_softplus(shape, beta, threshold):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref_out = F.softplus(x, beta=beta, threshold=threshold)
    tri_out = softplus(x, beta=beta, threshold=threshold)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
