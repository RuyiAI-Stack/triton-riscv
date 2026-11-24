import torch

from ._refdet import det3, spd
from .linalg_slogdet import linalg_slogdet


def test_linalg_slogdet():
    torch.manual_seed(0)
    a = spd()
    got = linalg_slogdet(a)
    ref = det3(a)
    torch.testing.assert_close(got.sign, torch.sign(ref), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        got.logabsdet, torch.log(torch.abs(ref)), rtol=1e-4, atol=1e-4
    )
