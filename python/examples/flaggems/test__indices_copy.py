import torch

from ._indices_copy import _indices_copy


def test__indices_copy():
    tensor = torch.eye(2).to_sparse()
    assert torch.equal(_indices_copy(tensor), tensor.indices())
