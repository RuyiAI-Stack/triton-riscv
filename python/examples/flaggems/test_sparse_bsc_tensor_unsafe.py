import torch

from .sparse_bsc_tensor_unsafe import sparse_bsc_tensor_unsafe


def test_sparse_bsc_tensor_unsafe():
    dense = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse_csc()
    rebuilt = sparse_bsc_tensor_unsafe(
        dense.ccol_indices(), dense.row_indices(), dense.values(), dense.shape
    )
    torch.testing.assert_close(
        rebuilt.to_dense(), dense.to_dense(), rtol=1e-4, atol=1e-4
    )
