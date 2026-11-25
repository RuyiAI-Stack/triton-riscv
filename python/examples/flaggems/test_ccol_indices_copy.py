import torch

from .ccol_indices_copy import ccol_indices_copy


def test_ccol_indices_copy():
    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]]).to_sparse_csc()
    assert torch.equal(ccol_indices_copy(dense), dense.ccol_indices())
