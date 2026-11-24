import torch

from .geqrf import geqrf
from .linalg_householder_product import linalg_householder_product


def test_linalg_householder_product():
    torch.manual_seed(0)
    g = torch.randn(3, 3)
    packed, tau = geqrf(g)
    q = linalg_householder_product(packed, tau)
    torch.testing.assert_close(q.T @ q, torch.eye(3), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(q @ torch.triu(packed), g, rtol=1e-4, atol=1e-4)
