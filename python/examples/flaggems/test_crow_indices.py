import torch

from .crow_indices import crow_indices


def test_crow_indices():
    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]]).to_sparse_csr()
    assert torch.equal(crow_indices(dense), dense.crow_indices())
