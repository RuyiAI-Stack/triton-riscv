import torch

from .ldl_factor_ex import ldl_factor_ex


def test_ldl_factor_ex():
    torch.manual_seed(0)
    raw = torch.randn(3, 3)
    a = raw @ raw.T + 0.5 * torch.eye(3)
    ld, pivots, info = ldl_factor_ex(a)
    assert int(info) == 0
    assert torch.equal(pivots, torch.arange(1, 4))
    unit = torch.tril(ld, diagonal=-1) + torch.eye(3)
    diag = torch.diag(ld.diagonal())
    torch.testing.assert_close(unit @ diag @ unit.T, a, rtol=1e-4, atol=1e-4)
