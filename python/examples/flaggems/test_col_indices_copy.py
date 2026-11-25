import torch

from .col_indices_copy import col_indices_copy


def test_col_indices_copy():
    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]]).to_sparse_csr()
    assert torch.equal(col_indices_copy(dense), dense.col_indices())
