import torch

from .nonzero_static import nonzero_static


def test_nonzero_static():
    values = torch.tensor([0.0, 2.0, 0.0, -1.0])
    assert nonzero_static(values).shape[0] == 2
