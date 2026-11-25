import torch

from ._linalg_slogdet import _linalg_slogdet
from ._refdet import det3, spd


def test__linalg_slogdet():
    torch.manual_seed(0)
    a = spd()
    got = _linalg_slogdet(a)
    ref = det3(a)
    torch.testing.assert_close(got.sign, torch.sign(ref), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        got.logabsdet, torch.log(torch.abs(ref)), rtol=1e-4, atol=1e-4
    )
