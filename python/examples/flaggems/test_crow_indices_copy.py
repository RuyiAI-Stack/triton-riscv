import torch

from .crow_indices_copy import crow_indices_copy


def test_crow_indices_copy():
    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]]).to_sparse_csr()
    assert torch.equal(crow_indices_copy(dense), dense.crow_indices())
