import torch

from .indices_copy import indices_copy


def test_indices_copy():
    tensor = torch.eye(2).to_sparse()
    assert torch.equal(indices_copy(tensor), tensor.indices())
