import torch

from .argsort import argsort


def test_argsort():
    values = torch.tensor([3.0, 1.0, 2.0])
    assert torch.equal(argsort(values), torch.argsort(values))
