import torch

from ._refdet import det3, spd
from .logdet import logdet


def test_logdet():
    torch.manual_seed(0)
    a = spd()
    torch.testing.assert_close(logdet(a), torch.log(det3(a)), rtol=1e-4, atol=1e-4)
