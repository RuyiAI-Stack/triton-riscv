import torch

from ._dirichlet_grad import _dirichlet_grad


def test__dirichlet_grad():
    sample = torch.tensor([[0.2, 0.8]])
    grad = _dirichlet_grad(sample, torch.ones_like(sample), torch.ones_like(sample))
    assert grad.shape == sample.shape
