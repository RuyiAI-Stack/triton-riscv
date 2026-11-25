import torch

from .geometric import geometric


def test_geometric():
    out = geometric(12, p=0.4)
    assert torch.all(out >= 1)
