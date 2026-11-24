import torch

from ._refdet import spd
from ._spsolve import _spsolve


def test__spsolve():
    torch.manual_seed(0)
    a = spd()
    b = torch.randn(3)
    got = _spsolve(a.to_sparse_csr(), b)
    torch.testing.assert_close(a @ got, b, rtol=1e-4, atol=1e-4)
