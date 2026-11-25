import torch

from .col_indices import col_indices


def test_col_indices():
    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]]).to_sparse_csr()
    assert torch.equal(col_indices(dense), dense.col_indices())
