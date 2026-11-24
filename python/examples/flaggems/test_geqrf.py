import torch

from .geqrf import geqrf
from .orgqr import orgqr


def test_geqrf():
    torch.manual_seed(0)
    g = torch.randn(3, 3)
    packed, tau = geqrf(g)
    q = orgqr(packed.clone(), tau)
    torch.testing.assert_close(q @ torch.triu(packed), g, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(q.T @ q, torch.eye(3), rtol=1e-4, atol=1e-4)
