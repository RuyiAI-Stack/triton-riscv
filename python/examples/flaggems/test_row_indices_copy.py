import torch

from .row_indices_copy import row_indices_copy


def test_row_indices_copy():
    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]]).to_sparse_csc()
    assert torch.equal(row_indices_copy(dense), dense.row_indices())
