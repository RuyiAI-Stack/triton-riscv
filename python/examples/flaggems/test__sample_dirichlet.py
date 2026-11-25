import torch

from ._sample_dirichlet import _sample_dirichlet


def test__sample_dirichlet():
    sample = _sample_dirichlet(torch.ones(2, 3))
    torch.testing.assert_close(sample.sum(-1), torch.ones(2), rtol=1e-4, atol=1e-5)
