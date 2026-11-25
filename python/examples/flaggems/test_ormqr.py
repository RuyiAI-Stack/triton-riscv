import torch

from .geqrf import geqrf
from .orgqr import orgqr
from .ormqr import ormqr


def test_ormqr():
    torch.manual_seed(0)
    g = torch.randn(3, 3)
    packed, tau = geqrf(g)
    other = torch.randn(3, 2)
    q = orgqr(packed.clone(), tau)
    got = ormqr(packed, tau, other, True, False)
    torch.testing.assert_close(got, q @ other, rtol=1e-4, atol=1e-4)
